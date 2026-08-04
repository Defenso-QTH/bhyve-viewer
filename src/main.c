/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c) 2026 Defenso
 */

/*
 * bhyve-view: present a bhyve guest's GPU output in a host Wayland window.
 *
 * The guest already renders on the host GPU -- with venus its images never
 * leave host GPU memory -- so bhyve exports the scanout as a dma_buf and
 * passes the fd over a unix socket.  This imports it with zwp_linux_dmabuf_v1
 * and hands it to the host compositor.  Nothing is read back, encoded or
 * compressed anywhere in that path.
 *
 * Keeping this out of bhyve is deliberate: the VMM stays free of Wayland and
 * EGL, and this can be jailed with nothing but the socket.
 */

#include <sys/socket.h>
#include <sys/un.h>

#include <errno.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <wayland-client.h>

#include "gpu_display.h"
#include "keymap.h"
#include "xdg-shell-client-protocol.h"
#include "linux-dmabuf-v1-client-protocol.h"

#define	MAX_BUFS	8

struct buf {
	uint32_t	id;
	bool		used;
	struct wl_buffer *wlbuf;
	uint32_t	width, height;
};

struct view {
	/* bhyve side */
	int		sock;
	uint8_t		inbuf[GPU_DISPLAY_MAX_MSG];
	size_t		inlen;

	/* wayland side */
	struct wl_display	*dpy;
	struct wl_registry	*registry;
	struct wl_compositor	*compositor;
	struct xdg_wm_base	*wm_base;
	struct zwp_linux_dmabuf_v1 *dmabuf;
	struct wl_seat		*seat;
	struct wl_keyboard	*kbd;
	struct wl_pointer	*ptr;
	struct wl_surface	*surface;
	struct xdg_surface	*xsurface;
	struct xdg_toplevel	*toplevel;

	struct buf	bufs[MAX_BUFS];
	uint32_t	cur_w, cur_h;	/* guest resolution, for pointer scaling */
	int32_t		win_w, win_h;	/* our window size, for pointer scaling */
	bool		configured;
	bool		running;
};

/* ------------------------------------------------------------------ */
/* bhyve socket							      */
/* ------------------------------------------------------------------ */

static bool
send_msg(struct view *v, const void *msg, size_t len)
{

	return (write(v->sock, msg, len) == (ssize_t)len);
}

static void
send_key(struct view *v, uint32_t evdev, bool down)
{
	struct gpu_display_key k;
	uint32_t xt = evdev_to_xt(evdev);

	if (xt == 0)
		return;		/* unmapped; bhyve's keysym path can't help here */

	memset(&k, 0, sizeof(k));
	k.hdr.type = GPU_DISPLAY_MSG_KEY;
	k.hdr.len = sizeof(k);
	k.down = down ? 1 : 0;
	k.keycode = xt;
	(void)send_msg(v, &k, sizeof(k));
}

static void
send_ptr(struct view *v, uint32_t buttons, int32_t x, int32_t y)
{
	struct gpu_display_ptr p;

	memset(&p, 0, sizeof(p));
	p.hdr.type = GPU_DISPLAY_MSG_PTR;
	p.hdr.len = sizeof(p);
	p.button = buttons;
	/*
	 * The guest's tablet is absolute in guest pixels, so scale from the
	 * window rather than forwarding raw surface coordinates -- otherwise
	 * the pointer drifts as soon as the window is not exactly the guest
	 * resolution.
	 */
	if (v->win_w > 0 && v->win_h > 0 && v->cur_w && v->cur_h) {
		p.x = (int32_t)((int64_t)x * v->cur_w / v->win_w);
		p.y = (int32_t)((int64_t)y * v->cur_h / v->win_h);
	} else {
		p.x = x;
		p.y = y;
	}
	(void)send_msg(v, &p, sizeof(p));
}

/* ------------------------------------------------------------------ */
/* buffers							      */
/* ------------------------------------------------------------------ */

static struct buf *
buf_find(struct view *v, uint32_t id)
{

	for (int i = 0; i < MAX_BUFS; i++)
		if (v->bufs[i].used && v->bufs[i].id == id)
			return (&v->bufs[i]);
	return (NULL);
}

static void
dmabuf_created(void *data, struct zwp_linux_buffer_params_v1 *params,
    struct wl_buffer *wlbuf)
{
	struct buf *b = data;

	b->wlbuf = wlbuf;
	zwp_linux_buffer_params_v1_destroy(params);
}

static void
dmabuf_failed(void *data, struct zwp_linux_buffer_params_v1 *params)
{
	struct buf *b = data;

	fprintf(stderr, "bhyve-view: compositor rejected dma_buf for buffer %u\n",
	    b->id);
	b->used = false;
	zwp_linux_buffer_params_v1_destroy(params);
}

static const struct zwp_linux_buffer_params_v1_listener params_listener = {
	.created = dmabuf_created,
	.failed = dmabuf_failed,
};

static void
handle_scanout(struct view *v, const struct gpu_display_scanout *so, int fd)
{
	struct zwp_linux_buffer_params_v1 *params;
	struct buf *b;
	int slot = -1;

	if (fd < 0) {
		fprintf(stderr, "bhyve-view: scanout %u arrived with no fd\n",
		    so->buffer_id);
		return;
	}
	if (so->transport != GPU_DISPLAY_XPORT_DMABUF) {
		fprintf(stderr, "bhyve-view: transport %u not supported yet\n",
		    so->transport);
		close(fd);
		return;
	}

	/* Replace the entry for this id, or take a free slot. */
	b = buf_find(v, so->buffer_id);
	if (b == NULL) {
		for (int i = 0; i < MAX_BUFS; i++)
			if (!v->bufs[i].used) { slot = i; break; }
		if (slot < 0) {
			fprintf(stderr, "bhyve-view: out of buffer slots\n");
			close(fd);
			return;
		}
		b = &v->bufs[slot];
	} else if (b->wlbuf != NULL) {
		wl_buffer_destroy(b->wlbuf);
		b->wlbuf = NULL;
	}

	b->id = so->buffer_id;
	b->width = so->width;
	b->height = so->height;
	b->used = true;
	v->cur_w = so->width;
	v->cur_h = so->height;

	params = zwp_linux_dmabuf_v1_create_params(v->dmabuf);
	zwp_linux_buffer_params_v1_add(params, fd, 0 /* plane */, 0 /* offset */,
	    so->stride, (uint32_t)(so->modifier >> 32),
	    (uint32_t)(so->modifier & 0xffffffff));
	zwp_linux_buffer_params_v1_add_listener(params, &params_listener, b);
	/*
	 * create_immed() would avoid a round trip, but it kills the client on
	 * an unimportable buffer instead of reporting it.  Given this is the
	 * first thing anyone will get wrong, take the asynchronous path and
	 * print a diagnostic.
	 */
	zwp_linux_buffer_params_v1_create(params, so->width, so->height,
	    so->drm_fourcc, 0 /* flags */);

	close(fd);	/* the compositor dup'd it */
}

static void
handle_frame(struct view *v, const struct gpu_display_frame *f)
{
	struct buf *b = buf_find(v, f->buffer_id);

	if (b == NULL || b->wlbuf == NULL)
		return;		/* not imported (yet); nothing to show */
	if (!v->configured)
		return;

	wl_surface_attach(v->surface, b->wlbuf, 0, 0);
	if (f->w != 0 && f->h != 0)
		wl_surface_damage_buffer(v->surface, (int32_t)f->x,
		    (int32_t)f->y, (int32_t)f->w, (int32_t)f->h);
	else
		wl_surface_damage_buffer(v->surface, 0, 0, (int32_t)b->width,
		    (int32_t)b->height);
	wl_surface_commit(v->surface);
}

/*
 * Read whatever is available and dispatch every complete message.  fds arrive
 * as ancillary data on the message that needs them, so recvmsg is used rather
 * than read.
 */
static bool
sock_readable(struct view *v)
{
	struct msghdr mh;
	struct iovec iov;
	union {
		struct cmsghdr hdr;
		unsigned char buf[CMSG_SPACE(sizeof(int))];
	} cmsgbuf;
	struct cmsghdr *cmsg;
	int fd = -1;
	ssize_t n;

	memset(&mh, 0, sizeof(mh));
	iov.iov_base = v->inbuf + v->inlen;
	iov.iov_len = sizeof(v->inbuf) - v->inlen;
	mh.msg_iov = &iov;
	mh.msg_iovlen = 1;
	mh.msg_control = cmsgbuf.buf;
	mh.msg_controllen = sizeof(cmsgbuf.buf);

	n = recvmsg(v->sock, &mh, 0);
	if (n <= 0)
		return (false);
	v->inlen += (size_t)n;

	for (cmsg = CMSG_FIRSTHDR(&mh); cmsg != NULL;
	    cmsg = CMSG_NXTHDR(&mh, cmsg)) {
		if (cmsg->cmsg_level == SOL_SOCKET &&
		    cmsg->cmsg_type == SCM_RIGHTS)
			memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
	}

	for (;;) {
		struct gpu_display_hdr hdr;

		if (v->inlen < sizeof(hdr))
			break;
		memcpy(&hdr, v->inbuf, sizeof(hdr));
		if (hdr.len < sizeof(hdr) || hdr.len > sizeof(v->inbuf)) {
			fprintf(stderr, "bhyve-view: bad message length %u\n",
			    hdr.len);
			return (false);
		}
		if (v->inlen < hdr.len)
			break;

		switch (hdr.type) {
		case GPU_DISPLAY_MSG_HELLO: {
			const struct gpu_display_hello *h = (const void *)v->inbuf;

			if (h->version != GPU_DISPLAY_VERSION)
				fprintf(stderr, "bhyve-view: protocol %u, "
				    "expected %u -- continuing anyway\n",
				    h->version, GPU_DISPLAY_VERSION);
			break;
		}
		case GPU_DISPLAY_MSG_SCANOUT:
			handle_scanout(v, (const void *)v->inbuf, fd);
			fd = -1;	/* consumed */
			break;
		case GPU_DISPLAY_MSG_FRAME:
			handle_frame(v, (const void *)v->inbuf);
			break;
		case GPU_DISPLAY_MSG_UNBIND:
			for (int i = 0; i < MAX_BUFS; i++) {
				if (v->bufs[i].wlbuf != NULL)
					wl_buffer_destroy(v->bufs[i].wlbuf);
				v->bufs[i].wlbuf = NULL;
				v->bufs[i].used = false;
			}
			break;
		default:
			break;	/* skip by length */
		}

		memmove(v->inbuf, v->inbuf + hdr.len, v->inlen - hdr.len);
		v->inlen -= hdr.len;
	}

	if (fd >= 0)
		close(fd);	/* arrived with a message that did not want it */
	return (true);
}

/* ------------------------------------------------------------------ */
/* wayland							      */
/* ------------------------------------------------------------------ */

static void
xdg_surface_configure(void *data, struct xdg_surface *s, uint32_t serial)
{
	struct view *v = data;

	xdg_surface_ack_configure(s, serial);
	v->configured = true;
}

static const struct xdg_surface_listener xdg_surface_listener = {
	.configure = xdg_surface_configure,
};

static void
toplevel_configure(void *data, struct xdg_toplevel *t __attribute__((unused)),
    int32_t w, int32_t h, struct wl_array *states __attribute__((unused)))
{
	struct view *v = data;

	if (w > 0 && h > 0) {
		v->win_w = w;
		v->win_h = h;
	}
}

static void
toplevel_close(void *data, struct xdg_toplevel *t __attribute__((unused)))
{
	struct view *v = data;

	v->running = false;
}

static const struct xdg_toplevel_listener toplevel_listener = {
	.configure = toplevel_configure,
	.close = toplevel_close,
};

static void
wm_base_ping(void *data __attribute__((unused)), struct xdg_wm_base *b,
    uint32_t serial)
{

	xdg_wm_base_pong(b, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	.ping = wm_base_ping,
};

/* --- input --- */

static void
kbd_key(void *data, struct wl_keyboard *k __attribute__((unused)),
    uint32_t serial __attribute__((unused)),
    uint32_t time __attribute__((unused)), uint32_t key, uint32_t state)
{
	struct view *v = data;

	/* wl_keyboard.key carries the evdev keycode directly. */
	send_key(v, key, state == WL_KEYBOARD_KEY_STATE_PRESSED);
}

static void kbd_keymap(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t f __attribute__((unused)), int32_t fd, uint32_t sz
    __attribute__((unused))) { if (fd >= 0) close(fd); }
static void kbd_enter(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused)),
    struct wl_array *ks __attribute__((unused))) {}
static void kbd_leave(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused))) {}
static void kbd_modifiers(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)), uint32_t md __attribute__((unused)),
    uint32_t ml __attribute__((unused)), uint32_t lk __attribute__((unused)),
    uint32_t g __attribute__((unused))) {}
static void kbd_repeat(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    int32_t r __attribute__((unused)), int32_t dl __attribute__((unused))) {}

static const struct wl_keyboard_listener kbd_listener = {
	.keymap = kbd_keymap, .enter = kbd_enter, .leave = kbd_leave,
	.key = kbd_key, .modifiers = kbd_modifiers,
	.repeat_info = kbd_repeat,
};

static void
ptr_motion(void *data, struct wl_pointer *p __attribute__((unused)),
    uint32_t time __attribute__((unused)), wl_fixed_t sx, wl_fixed_t sy)
{
	struct view *v = data;
	static uint32_t buttons;

	send_ptr(v, buttons, wl_fixed_to_int(sx), wl_fixed_to_int(sy));
}

static void
ptr_button(void *data, struct wl_pointer *p __attribute__((unused)),
    uint32_t serial __attribute__((unused)),
    uint32_t time __attribute__((unused)), uint32_t button, uint32_t state)
{
	struct view *v = data;
	static uint32_t buttons;
	uint32_t bit;

	/* BTN_LEFT/RIGHT/MIDDLE -> the mask console_ptr_event expects. */
	switch (button) {
	case 0x110: bit = 0x01; break;	/* BTN_LEFT */
	case 0x111: bit = 0x04; break;	/* BTN_RIGHT */
	case 0x112: bit = 0x02; break;	/* BTN_MIDDLE */
	default: return;
	}
	if (state)
		buttons |= bit;
	else
		buttons &= ~bit;
	send_ptr(v, buttons, 0, 0);
}

static void ptr_enter(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused)),
    wl_fixed_t x __attribute__((unused)),
    wl_fixed_t y __attribute__((unused))) {}
static void ptr_leave(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused))) {}
static void ptr_axis(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t t __attribute__((unused)), uint32_t a __attribute__((unused)),
    wl_fixed_t val __attribute__((unused))) {}
static void ptr_frame(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused))) {}
static void ptr_axis_src(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused))) {}
static void ptr_axis_stop(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t t __attribute__((unused)), uint32_t a __attribute__((unused))) {}
static void ptr_axis_disc(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t a __attribute__((unused)), int32_t dc __attribute__((unused))) {}

static const struct wl_pointer_listener ptr_listener = {
	.enter = ptr_enter, .leave = ptr_leave, .motion = ptr_motion,
	.button = ptr_button, .axis = ptr_axis, .frame = ptr_frame,
	.axis_source = ptr_axis_src, .axis_stop = ptr_axis_stop,
	.axis_discrete = ptr_axis_disc,
};

static void
seat_caps(void *data, struct wl_seat *seat, uint32_t caps)
{
	struct view *v = data;

	if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && v->kbd == NULL) {
		v->kbd = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(v->kbd, &kbd_listener, v);
	}
	if ((caps & WL_SEAT_CAPABILITY_POINTER) && v->ptr == NULL) {
		v->ptr = wl_seat_get_pointer(seat);
		wl_pointer_add_listener(v->ptr, &ptr_listener, v);
	}
}

static void seat_name(void *d __attribute__((unused)),
    struct wl_seat *s __attribute__((unused)),
    const char *n __attribute__((unused))) {}

static const struct wl_seat_listener seat_listener = {
	.capabilities = seat_caps, .name = seat_name,
};

static void
registry_global(void *data, struct wl_registry *reg, uint32_t name,
    const char *iface, uint32_t version)
{
	struct view *v = data;

	if (strcmp(iface, wl_compositor_interface.name) == 0)
		v->compositor = wl_registry_bind(reg, name,
		    &wl_compositor_interface, 4);
	else if (strcmp(iface, xdg_wm_base_interface.name) == 0) {
		v->wm_base = wl_registry_bind(reg, name,
		    &xdg_wm_base_interface, 1);
		xdg_wm_base_add_listener(v->wm_base, &wm_base_listener, v);
	} else if (strcmp(iface, zwp_linux_dmabuf_v1_interface.name) == 0)
		v->dmabuf = wl_registry_bind(reg, name,
		    &zwp_linux_dmabuf_v1_interface, version < 3 ? version : 3);
	else if (strcmp(iface, wl_seat_interface.name) == 0) {
		v->seat = wl_registry_bind(reg, name, &wl_seat_interface, 5);
		wl_seat_add_listener(v->seat, &seat_listener, v);
	}
}

static void registry_remove(void *d __attribute__((unused)),
    struct wl_registry *r __attribute__((unused)),
    uint32_t n __attribute__((unused))) {}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global, .global_remove = registry_remove,
};

/* ------------------------------------------------------------------ */

static int
connect_socket(const char *path)
{
	struct sockaddr_un sun;
	int fd;

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0)
		return (-1);
	memset(&sun, 0, sizeof(sun));
	sun.sun_family = AF_UNIX;
	strncpy(sun.sun_path, path, sizeof(sun.sun_path) - 1);
	if (connect(fd, (struct sockaddr *)&sun, sizeof(sun)) != 0) {
		close(fd);
		return (-1);
	}
	return (fd);
}

int
main(int argc, char **argv)
{
	struct view v;
	struct pollfd pfd[2];

	if (argc != 2) {
		fprintf(stderr, "usage: bhyve-view <socket-path>\n");
		return (1);
	}

	memset(&v, 0, sizeof(v));
	v.running = true;
	v.win_w = 1920;
	v.win_h = 1080;

	v.sock = connect_socket(argv[1]);
	if (v.sock < 0) {
		fprintf(stderr, "bhyve-view: cannot connect to %s: %s\n",
		    argv[1], strerror(errno));
		return (1);
	}

	v.dpy = wl_display_connect(NULL);
	if (v.dpy == NULL) {
		fprintf(stderr, "bhyve-view: no Wayland display "
		    "(is WAYLAND_DISPLAY set?)\n");
		return (1);
	}
	v.registry = wl_display_get_registry(v.dpy);
	wl_registry_add_listener(v.registry, &registry_listener, &v);
	wl_display_roundtrip(v.dpy);

	if (v.compositor == NULL || v.wm_base == NULL || v.dmabuf == NULL) {
		fprintf(stderr, "bhyve-view: compositor lacks %s\n",
		    v.dmabuf == NULL ? "zwp_linux_dmabuf_v1" :
		    "wl_compositor/xdg_wm_base");
		return (1);
	}

	v.surface = wl_compositor_create_surface(v.compositor);
	v.xsurface = xdg_wm_base_get_xdg_surface(v.wm_base, v.surface);
	xdg_surface_add_listener(v.xsurface, &xdg_surface_listener, &v);
	v.toplevel = xdg_surface_get_toplevel(v.xsurface);
	xdg_toplevel_add_listener(v.toplevel, &toplevel_listener, &v);
	xdg_toplevel_set_title(v.toplevel, "bhyve");
	wl_surface_commit(v.surface);
	wl_display_roundtrip(v.dpy);

	while (v.running) {
		wl_display_flush(v.dpy);

		pfd[0].fd = wl_display_get_fd(v.dpy);
		pfd[0].events = POLLIN;
		pfd[1].fd = v.sock;
		pfd[1].events = POLLIN;

		if (poll(pfd, 2, -1) < 0) {
			if (errno == EINTR)
				continue;
			break;
		}
		if (pfd[0].revents & POLLIN) {
			if (wl_display_dispatch(v.dpy) < 0)
				break;
		}
		if (pfd[1].revents & (POLLIN | POLLHUP)) {
			if (!sock_readable(&v)) {
				fprintf(stderr, "bhyve-view: bhyve closed the "
				    "connection\n");
				break;
			}
		}
	}

	wl_display_disconnect(v.dpy);
	close(v.sock);
	return (0);
}
