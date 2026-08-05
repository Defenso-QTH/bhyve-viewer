/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c) 2026 Defenso
 */

/*
 * bhyve-viewer: present a bhyve guest's GPU output in a host Wayland window.
 *
 * The guest already renders on the host GPU -- with venus its images never
 * leave host GPU memory -- so bhyve exports the scanout as a dma_buf and
 * passes the fd over a unix socket.
 *
 * The buffer is imported through EGL (EGL_LINUX_DMA_BUF_EXT) and drawn to an
 * ordinary Wayland surface, rather than handed to the compositor as a
 * linux-dmabuf wl_buffer.  That is deliberate.  Handing it over directly
 * requires telling the compositor the buffer's format modifier, and
 * virglrenderer does not report one: resource_get_info_ext() returns
 * DRM_FORMAT_MOD_INVALID whatever the guest allocates.  Asserting a layout we
 * do not know produced periodic corruption -- the compositor reading tiled or
 * DCC-compressed memory as raw pixels -- and no guessed value fixed it,
 * because the information was never available to guess from.
 *
 * Importing through EGL sidesteps the question: Mesa allocated these buffers
 * and Mesa reads them back, on the same GPU with the same driver, so the
 * implicit layout is known to both ends.  The cost is one GPU-to-GPU blit per
 * frame instead of a direct handover -- still no CPU copy and no encode.
 *
 * Keeping this out of bhyve is deliberate too: the VMM stays free of EGL and
 * Wayland, and this can be jailed with nothing but the socket.
 */

#include <sys/socket.h>
#include <sys/un.h>

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <wayland-client.h>
#include <wayland-egl.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "gpu_display.h"
#include "keymap.h"
#include "xdg-shell-client-protocol.h"

#define	MAX_BUFS	8

/*
 * Overrides kept from the linux-dmabuf implementation this replaced: the
 * values bhyve reports proved unreliable, and being able to try another
 * without a rebuild saved several cycles.  The modifier override is gone --
 * with an implicit EGL import there is no modifier to override.
 */
static uint32_t	opt_fourcc;
static int32_t	opt_offset = -1;
static bool	opt_flip;
static bool	opt_verbose;	/* -v: report every input event, not a sample */

struct buf {
	uint32_t	id;
	bool		used;
	EGLImageKHR	image;
	GLuint		tex;
	uint32_t	width, height;
};

struct view {
	int		sock;
	uint8_t		inbuf[GPU_DISPLAY_MAX_MSG];
	size_t		inlen;

	struct wl_display	*dpy;
	struct wl_registry	*registry;
	struct wl_compositor	*compositor;
	struct xdg_wm_base	*wm_base;
	struct wl_seat		*seat;
	struct wl_keyboard	*kbd;
	struct wl_pointer	*ptr;
	struct wl_surface	*surface;
	struct xdg_surface	*xsurface;
	struct xdg_toplevel	*toplevel;
	struct wl_egl_window	*egl_window;

	EGLDisplay	egl_dpy;
	EGLContext	egl_ctx;
	EGLSurface	egl_surf;
	EGLConfig	egl_cfg;

	GLuint		prog;
	GLint		attr_pos;
	GLint		uni_tex;
	GLint		uni_flip;

	struct buf	bufs[MAX_BUFS];
	uint32_t	cur_w, cur_h;
	int32_t		win_w, win_h;
	bool		configured;
	bool		running;
	/*
	 * Newest buffer the guest has flipped to, drawn once per loop pass.
	 * The guest flips at its own rate; drawing every flip means blocking
	 * in eglSwapBuffers as often as the guest presents, which starves the
	 * Wayland event loop and stops input being dispatched at all.
	 */
	uint32_t	pending_buf;
	bool		have_pending;
	int		pending_fence;	/* sync_file for that buffer, or -1 */
};

static PFNEGLCREATESYNCKHRPROC		p_eglCreateSyncKHR;
static PFNEGLDESTROYSYNCKHRPROC		p_eglDestroySyncKHR;
static PFNEGLWAITSYNCKHRPROC		p_eglWaitSyncKHR;
static PFNEGLCREATEIMAGEKHRPROC		p_eglCreateImageKHR;
static PFNEGLDESTROYIMAGEKHRPROC	p_eglDestroyImageKHR;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES;

/* ------------------------------------------------------------------ */
/* bhyve socket							      */
/* ------------------------------------------------------------------ */

static bool
send_msg(struct view *v, const void *msg, size_t len)
{

	return (write(v->sock, msg, len) == (ssize_t)len);
}

static uint64_t keys_sent, ptrs_sent;

/*
 * Which evdev keys we have told the guest are down.
 *
 * Wayland stops delivering to a surface that loses focus, so a key pressed
 * before the focus change never gets its release -- and the guest is left
 * holding it.  A stuck modifier is the visible form of this, and because the
 * emulated keyboard's state is shared by every input source it also breaks
 * bhyve's own VNC, not just this viewer.
 */
static uint8_t	key_is_down[256 / 8];

/*
 * Set by SIGINT/SIGTERM so the main loop exits by its normal path, which
 * releases any keys the guest still thinks are down.  Quitting without that
 * leaves a stuck modifier behind, and since the emulated keyboard is shared it
 * breaks every other input path until something happens to clear it.
 * SIGKILL cannot be caught, so `kill -9` will still strand them.
 */
static volatile sig_atomic_t quit_requested;

static void
on_signal(int sig __attribute__((unused)))
{

	quit_requested = 1;
}

static void
key_mark(uint32_t evdev, bool down)
{

	if (evdev >= 256)
		return;
	if (down)
		key_is_down[evdev / 8] |= 1u << (evdev % 8);
	else
		key_is_down[evdev / 8] &= ~(1u << (evdev % 8));
}

static void
send_key(struct view *v, uint32_t evdev, bool down)
{
	struct gpu_display_key k;
	uint32_t xt = evdev_to_xt(evdev);

	if (xt == 0) {
		fprintf(stderr, "bhyve-viewer: key evdev=%u has no XT mapping, "
		    "dropped\n", evdev);
		return;
	}
	memset(&k, 0, sizeof(k));
	k.hdr.type = GPU_DISPLAY_MSG_KEY;
	k.hdr.len = sizeof(k);
	k.down = down ? 1 : 0;
	k.keycode = xt;
	key_mark(evdev, down);
	if (!send_msg(v, &k, sizeof(k)))
		fprintf(stderr, "bhyve-viewer: key send failed: %s\n",
		    strerror(errno));
	else if (opt_verbose || keys_sent < 5)
		fprintf(stderr, "bhyve-viewer: sent key #%ju evdev=%u xt=0x%x "
		    "%s\n", (uintmax_t)keys_sent + 1, evdev, xt,
		    down ? "down" : "up");
	keys_sent++;
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
	 * The guest tablet is absolute in guest pixels, so scale from the
	 * window rather than forwarding surface coordinates -- otherwise the
	 * pointer drifts whenever the window is not the guest resolution.
	 */
	if (v->win_w > 0 && v->win_h > 0 && v->cur_w && v->cur_h) {
		p.x = (int32_t)((int64_t)x * v->cur_w / v->win_w);
		p.y = (int32_t)((int64_t)y * v->cur_h / v->win_h);
	} else {
		p.x = x;
		p.y = y;
	}
	if (!send_msg(v, &p, sizeof(p)))
		fprintf(stderr, "bhyve-viewer: ptr send failed: %s\n",
		    strerror(errno));
	else if (opt_verbose || ptrs_sent < 5)
		fprintf(stderr, "bhyve-viewer: sent ptr #%ju buttons=0x%x "
		    "%d,%d\n", (uintmax_t)ptrs_sent + 1, buttons, p.x, p.y);
	ptrs_sent++;
}

/* ------------------------------------------------------------------ */
/* GL								      */
/* ------------------------------------------------------------------ */

/*
 * A dma_buf imported through EGL is sampled with samplerExternalOES, not
 * sampler2D.  The external target is what lets the driver apply whatever
 * tiling or compression the buffer actually carries, which is the entire
 * reason this path works where naming a modifier did not.
 */
static const char *vert_src =
"attribute vec2 pos;\n"
"varying vec2 uv;\n"
"uniform float flip;\n"
"void main() {\n"
"  uv = vec2((pos.x + 1.0) * 0.5, (1.0 - flip * pos.y) * 0.5);\n"
"  gl_Position = vec4(pos, 0.0, 1.0);\n"
"}\n";

static const char *frag_src =
"#extension GL_OES_EGL_image_external : require\n"
"precision mediump float;\n"
"varying vec2 uv;\n"
"uniform samplerExternalOES tex;\n"
"void main() { gl_FragColor = texture2D(tex, uv); }\n";

static GLuint
compile(GLenum type, const char *src)
{
	GLuint s = glCreateShader(type);
	GLint ok = 0;

	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[512];

		glGetShaderInfoLog(s, sizeof(log), NULL, log);
		fprintf(stderr, "bhyve-viewer: shader: %s\n", log);
		return (0);
	}
	return (s);
}

static bool
gl_setup(struct view *v)
{
	GLuint vs = compile(GL_VERTEX_SHADER, vert_src);
	GLuint fs = compile(GL_FRAGMENT_SHADER, frag_src);
	GLint ok = 0;

	if (vs == 0 || fs == 0)
		return (false);
	v->prog = glCreateProgram();
	glAttachShader(v->prog, vs);
	glAttachShader(v->prog, fs);
	glLinkProgram(v->prog);
	glGetProgramiv(v->prog, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[512];

		glGetProgramInfoLog(v->prog, sizeof(log), NULL, log);
		fprintf(stderr, "bhyve-viewer: link: %s\n", log);
		return (false);
	}
	v->attr_pos = glGetAttribLocation(v->prog, "pos");
	v->uni_tex = glGetUniformLocation(v->prog, "tex");
	v->uni_flip = glGetUniformLocation(v->prog, "flip");
	return (true);
}

static void
draw(struct view *v, struct buf *b, int fence_fd)
{
	static const GLfloat quad[] = {
		-1.f, -1.f,  1.f, -1.f, -1.f, 1.f,
		 1.f, -1.f,  1.f,  1.f, -1.f, 1.f,
	};

	/*
	 * Make the GPU wait until the guest has finished drawing into this
	 * buffer.  eglWaitSyncKHR queues the wait on the GL command stream
	 * rather than blocking here, so the cost is ordering, not a stall.
	 * Without it we sample whatever is in the buffer at the moment we
	 * happen to read it, which on a fast renderer is part of one frame
	 * and part of the next.
	 */
	if (fence_fd >= 0 && p_eglCreateSyncKHR != NULL) {
		static uintmax_t waits;
		EGLint attrs[] = {
			EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fence_fd, EGL_NONE
		};
		EGLSyncKHR sync = p_eglCreateSyncKHR(v->egl_dpy,
		    EGL_SYNC_NATIVE_FENCE_ANDROID, attrs);

		if (sync != EGL_NO_SYNC_KHR) {
			/* eglCreateSyncKHR took the fd. */
			p_eglWaitSyncKHR(v->egl_dpy, sync, 0);
			p_eglDestroySyncKHR(v->egl_dpy, sync);
			if (opt_verbose || waits < 3)
				fprintf(stderr, "bhyve-viewer: waited on "
				    "fence #%ju\n", (uintmax_t)waits + 1);
			waits++;
		} else {
			fprintf(stderr, "bhyve-viewer: eglCreateSyncKHR "
			    "failed (0x%x); drawing unsynchronised\n",
			    eglGetError()); {
			close(fence_fd);
		}
	} else if (fence_fd >= 0)
		close(fence_fd);

	glViewport(0, 0, v->win_w, v->win_h);
	glUseProgram(v->prog);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_EXTERNAL_OES, b->tex);
	glUniform1i(v->uni_tex, 0);
	glUniform1f(v->uni_flip, opt_flip ? -1.f : 1.f);
	glVertexAttribPointer(v->attr_pos, 2, GL_FLOAT, GL_FALSE, 0, quad);
	glEnableVertexAttribArray(v->attr_pos);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	eglSwapBuffers(v->egl_dpy, v->egl_surf);
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
buf_release(struct view *v, struct buf *b)
{

	if (b->image != EGL_NO_IMAGE_KHR && b->image != NULL)
		p_eglDestroyImageKHR(v->egl_dpy, b->image);
	if (b->tex != 0)
		glDeleteTextures(1, &b->tex);
	memset(b, 0, sizeof(*b));
}

static void
handle_scanout(struct view *v, const struct gpu_display_scanout *so, int fd)
{
	struct buf *b;
	uint32_t fourcc = opt_fourcc ? opt_fourcc : so->drm_fourcc;
	uint32_t off = opt_offset >= 0 ? (uint32_t)opt_offset : so->offset;
	int slot = -1;
	EGLint attrs[] = {
		EGL_WIDTH,			(EGLint)so->width,
		EGL_HEIGHT,			(EGLint)so->height,
		EGL_LINUX_DRM_FOURCC_EXT,	(EGLint)fourcc,
		EGL_DMA_BUF_PLANE0_FD_EXT,	fd,
		EGL_DMA_BUF_PLANE0_OFFSET_EXT,	(EGLint)off,
		EGL_DMA_BUF_PLANE0_PITCH_EXT,	(EGLint)so->stride,
		EGL_NONE
		/*
		 * Deliberately no EGL_DMA_BUF_PLANE0_MODIFIER_*_EXT.  Omitting
		 * them requests an implicit-modifier import: Mesa allocated
		 * this buffer and knows its layout, and not having to name one
		 * is the whole point of this path.
		 */
	};

	if (fd < 0) {
		fprintf(stderr, "bhyve-viewer: scanout %u arrived with no fd\n",
		    so->buffer_id);
		return;
	}
	if (so->transport != GPU_DISPLAY_XPORT_DMABUF) {
		fprintf(stderr, "bhyve-viewer: transport %u unsupported\n",
		    so->transport);
		close(fd);
		return;
	}

	b = buf_find(v, so->buffer_id);
	if (b == NULL) {
		for (int i = 0; i < MAX_BUFS; i++)
			if (!v->bufs[i].used) {
				slot = i;
				break;
			}
		if (slot < 0) {
			fprintf(stderr, "bhyve-viewer: out of buffer slots\n");
			close(fd);
			return;
		}
		b = &v->bufs[slot];
	} else
		buf_release(v, b);

	fprintf(stderr, "bhyve-viewer: import %ux%u fourcc=0x%08x stride=%u "
	    "offset=%u (implicit modifier)\n", so->width, so->height, fourcc,
	    so->stride, off);

	b->image = p_eglCreateImageKHR(v->egl_dpy, EGL_NO_CONTEXT,
	    EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
	if (b->image == EGL_NO_IMAGE_KHR) {
		fprintf(stderr, "bhyve-viewer: eglCreateImageKHR failed "
		    "(0x%x)\n", eglGetError());
		close(fd);
		return;
	}

	glGenTextures(1, &b->tex);
	glBindTexture(GL_TEXTURE_EXTERNAL_OES, b->tex);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER,
	    GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER,
	    GL_LINEAR);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S,
	    GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T,
	    GL_CLAMP_TO_EDGE);
	p_glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, b->image);

	b->id = so->buffer_id;
	b->width = so->width;
	b->height = so->height;
	b->used = true;
	v->cur_w = so->width;
	v->cur_h = so->height;

	/* EGL holds its own reference to the underlying buffer. */
	close(fd);
}

static void
handle_frame(struct view *v, const struct gpu_display_frame *f, int fence_fd)
{
	struct buf *b = buf_find(v, f->buffer_id);

	if (b == NULL || b->tex == 0 || !v->configured) {
		if (fence_fd >= 0)
			close(fence_fd);
		return;
	}
	/*
	 * Coalesce: only the most recent flip is worth drawing, and only its
	 * fence matters -- a superseded frame's fence guards a buffer we are
	 * no longer going to read.
	 */
	if (v->pending_fence >= 0)
		close(v->pending_fence);
	v->pending_buf = f->buffer_id;
	v->pending_fence = fence_fd;
	v->have_pending = true;
}

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
	    cmsg = CMSG_NXTHDR(&mh, cmsg))
		if (cmsg->cmsg_level == SOL_SOCKET &&
		    cmsg->cmsg_type == SCM_RIGHTS)
			memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));

	for (;;) {
		struct gpu_display_hdr hdr;

		if (v->inlen < sizeof(hdr))
			break;
		memcpy(&hdr, v->inbuf, sizeof(hdr));
		if (hdr.len < sizeof(hdr) || hdr.len > sizeof(v->inbuf)) {
			fprintf(stderr, "bhyve-viewer: bad length %u\n",
			    hdr.len);
			return (false);
		}
		if (v->inlen < hdr.len)
			break;

		switch (hdr.type) {
		case GPU_DISPLAY_MSG_HELLO: {
			const struct gpu_display_hello *h =
			    (const void *)v->inbuf;

			if (h->version != GPU_DISPLAY_VERSION)
				fprintf(stderr, "bhyve-viewer: protocol %u vs "
				    "%u\n", h->version, GPU_DISPLAY_VERSION);
			break;
		}
		case GPU_DISPLAY_MSG_SCANOUT:
			handle_scanout(v, (const void *)v->inbuf, fd);
			fd = -1;
			break;
		case GPU_DISPLAY_MSG_FRAME: {
			const struct gpu_display_frame *f = (const void *)v->inbuf;

			if (!f->has_fence) {
				static uintmax_t unfenced;

				if (opt_verbose || unfenced < 3)
					fprintf(stderr, "bhyve-viewer: frame "
					    "#%ju arrived with no fence\n",
					    (uintmax_t)unfenced + 1);
				unfenced++;
			}
			handle_frame(v, f, f->has_fence ? fd : -1);
			if (f->has_fence)
				fd = -1;	/* consumed */
			break;
		}
		case GPU_DISPLAY_MSG_UNBIND:
			for (int i = 0; i < MAX_BUFS; i++)
				if (v->bufs[i].used)
					buf_release(v, &v->bufs[i]);
			break;
		default:
			break;
		}
		memmove(v->inbuf, v->inbuf + hdr.len, v->inlen - hdr.len);
		v->inlen -= hdr.len;
	}

	if (fd >= 0)
		close(fd);
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
    int32_t w, int32_t h, struct wl_array *st __attribute__((unused)))
{
	struct view *v = data;

	if (w > 0 && h > 0) {
		v->win_w = w;
		v->win_h = h;
		if (v->egl_window != NULL)
			wl_egl_window_resize(v->egl_window, w, h, 0, 0);
	}
}

static void
toplevel_close(void *data, struct xdg_toplevel *t __attribute__((unused)))
{

	((struct view *)data)->running = false;
}

static const struct xdg_toplevel_listener toplevel_listener = {
	.configure = toplevel_configure, .close = toplevel_close,
};

static void
wm_base_ping(void *d __attribute__((unused)), struct xdg_wm_base *b,
    uint32_t serial)
{

	xdg_wm_base_pong(b, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	.ping = wm_base_ping,
};

static void
kbd_key(void *data, struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)), uint32_t t __attribute__((unused)),
    uint32_t key, uint32_t state)
{

	/* wl_keyboard.key carries the evdev keycode directly. */
	send_key(data, key, state == WL_KEYBOARD_KEY_STATE_PRESSED);
}

static void kbd_keymap(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t f __attribute__((unused)), int32_t fd,
    uint32_t sz __attribute__((unused))) { if (fd >= 0) close(fd); }
static void kbd_enter(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused)),
    struct wl_array *ks __attribute__((unused)))
{ fprintf(stderr, "bhyve-viewer: keyboard focus gained\n"); }
static void
kbd_leave(void *data, struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused)))
{
	struct view *v = data;
	unsigned n = 0;

	/*
	 * Release anything still held.  Without this a modifier pressed as
	 * focus moves away stays down in the guest for good, and since the
	 * emulated keyboard is shared it breaks every other input path too.
	 */
	for (uint32_t code = 0; code < 256; code++)
		if (key_is_down[code / 8] & (1u << (code % 8))) {
			send_key(v, code, false);
			n++;
		}
	fprintf(stderr, "bhyve-viewer: keyboard focus lost%s\n",
	    n ? " (released held keys)" : "");
}
static void kbd_mods(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    uint32_t s __attribute__((unused)), uint32_t a __attribute__((unused)),
    uint32_t b __attribute__((unused)), uint32_t c __attribute__((unused)),
    uint32_t g __attribute__((unused))) {}
static void kbd_rep(void *d __attribute__((unused)),
    struct wl_keyboard *k __attribute__((unused)),
    int32_t r __attribute__((unused)), int32_t dl __attribute__((unused))) {}

static const struct wl_keyboard_listener kbd_listener = {
	.keymap = kbd_keymap, .enter = kbd_enter, .leave = kbd_leave,
	.key = kbd_key, .modifiers = kbd_mods, .repeat_info = kbd_rep,
};

static uint32_t ptr_buttons;

static void
ptr_motion(void *data, struct wl_pointer *p __attribute__((unused)),
    uint32_t t __attribute__((unused)), wl_fixed_t sx, wl_fixed_t sy)
{

	send_ptr(data, ptr_buttons, wl_fixed_to_int(sx), wl_fixed_to_int(sy));
}

static void
ptr_button(void *data, struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused)), uint32_t t __attribute__((unused)),
    uint32_t button, uint32_t state)
{
	uint32_t bit;

	switch (button) {
	case 0x110: bit = 0x01; break;	/* BTN_LEFT */
	case 0x111: bit = 0x04; break;	/* BTN_RIGHT */
	case 0x112: bit = 0x02; break;	/* BTN_MIDDLE */
	default: return;
	}
	if (state)
		ptr_buttons |= bit;
	else
		ptr_buttons &= ~bit;
	send_ptr(data, ptr_buttons, 0, 0);
}

static void ptr_enter(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused)),
    struct wl_surface *su __attribute__((unused)),
    wl_fixed_t x __attribute__((unused)),
    wl_fixed_t y __attribute__((unused)))
{ fprintf(stderr, "bhyve-viewer: pointer entered\n"); }
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
static void ptr_asrc(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t s __attribute__((unused))) {}
static void ptr_astop(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t t __attribute__((unused)), uint32_t a __attribute__((unused))) {}
static void ptr_adisc(void *d __attribute__((unused)),
    struct wl_pointer *p __attribute__((unused)),
    uint32_t a __attribute__((unused)), int32_t dc __attribute__((unused))) {}

static const struct wl_pointer_listener ptr_listener = {
	.enter = ptr_enter, .leave = ptr_leave, .motion = ptr_motion,
	.button = ptr_button, .axis = ptr_axis, .frame = ptr_frame,
	.axis_source = ptr_asrc, .axis_stop = ptr_astop,
	.axis_discrete = ptr_adisc,
};

static void
seat_caps(void *data, struct wl_seat *seat, uint32_t caps)
{
	struct view *v = data;

	fprintf(stderr, "bhyve-viewer: seat caps=0x%x (keyboard=%s pointer=%s)\n",
	    caps, (caps & WL_SEAT_CAPABILITY_KEYBOARD) ? "yes" : "no",
	    (caps & WL_SEAT_CAPABILITY_POINTER) ? "yes" : "no");
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
    const char *iface, uint32_t version __attribute__((unused)))
{
	struct view *v = data;

	if (strcmp(iface, wl_compositor_interface.name) == 0)
		v->compositor = wl_registry_bind(reg, name,
		    &wl_compositor_interface, 4);
	else if (strcmp(iface, xdg_wm_base_interface.name) == 0) {
		v->wm_base = wl_registry_bind(reg, name,
		    &xdg_wm_base_interface, 1);
		xdg_wm_base_add_listener(v->wm_base, &wm_base_listener, v);
	} else if (strcmp(iface, wl_seat_interface.name) == 0) {
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

static bool
egl_setup(struct view *v)
{
	static const EGLint cfg_attrs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
		EGL_NONE
	};
	static const EGLint ctx_attrs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE
	};
	EGLint n = 0, major, minor;

	v->egl_dpy = eglGetDisplay((EGLNativeDisplayType)v->dpy);
	if (v->egl_dpy == EGL_NO_DISPLAY ||
	    !eglInitialize(v->egl_dpy, &major, &minor)) {
		fprintf(stderr, "bhyve-viewer: eglInitialize failed\n");
		return (false);
	}
	if (!eglBindAPI(EGL_OPENGL_ES_API) ||
	    !eglChooseConfig(v->egl_dpy, cfg_attrs, &v->egl_cfg, 1, &n) ||
	    n == 0) {
		fprintf(stderr, "bhyve-viewer: no usable EGL config\n");
		return (false);
	}

	p_eglCreateSyncKHR = (void *)eglGetProcAddress("eglCreateSyncKHR");
	p_eglDestroySyncKHR = (void *)eglGetProcAddress("eglDestroySyncKHR");
	p_eglWaitSyncKHR = (void *)eglGetProcAddress("eglWaitSyncKHR");
	if (p_eglCreateSyncKHR == NULL || p_eglWaitSyncKHR == NULL)
		fprintf(stderr, "bhyve-viewer: no EGL_ANDROID_native_fence_sync;"
		    " frames will not be synchronised with the guest\n");
	p_eglCreateImageKHR = (void *)eglGetProcAddress("eglCreateImageKHR");
	p_eglDestroyImageKHR = (void *)eglGetProcAddress("eglDestroyImageKHR");
	p_glEGLImageTargetTexture2DOES =
	    (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");
	if (p_eglCreateImageKHR == NULL || p_eglDestroyImageKHR == NULL ||
	    p_glEGLImageTargetTexture2DOES == NULL) {
		fprintf(stderr, "bhyve-viewer: EGL lacks dma_buf image import "
		    "(needs EGL_EXT_image_dma_buf_import)\n");
		return (false);
	}

	v->egl_ctx = eglCreateContext(v->egl_dpy, v->egl_cfg, EGL_NO_CONTEXT,
	    ctx_attrs);
	if (v->egl_ctx == EGL_NO_CONTEXT) {
		fprintf(stderr, "bhyve-viewer: eglCreateContext failed\n");
		return (false);
	}
	return (true);
}

static int
connect_socket(const char *path)
{
	struct sockaddr_un sun;
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);

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
	int c;

	memset(&v, 0, sizeof(v));
	v.running = true;
	v.pending_fence = -1;
	v.win_w = 1920;
	v.win_h = 1080;

	while ((c = getopt(argc, argv, "f:o:Fv")) != -1) {
		switch (c) {
		case 'f':
			if (strlen(optarg) == 4)
				opt_fourcc = (uint32_t)optarg[0] |
				    ((uint32_t)optarg[1] << 8) |
				    ((uint32_t)optarg[2] << 16) |
				    ((uint32_t)optarg[3] << 24);
			else
				opt_fourcc = (uint32_t)strtoul(optarg, NULL, 0);
			break;
		case 'o':
			opt_offset = (int32_t)strtol(optarg, NULL, 0);
			break;
		case 'F':
			opt_flip = true;
			break;
		case 'v':
			opt_verbose = true;
			break;
		default:
			goto usage;
		}
	}
	argc -= optind;
	argv += optind;
	if (argc != 1) {
usage:
		fprintf(stderr, "usage: bhyve-viewer [-f fourcc] [-o offset] "
		    "[-F] <socket-path>\n"
		    "  -f  four character code, e.g. XR24\n"
		    "  -o  byte offset of plane 0\n"
		    "  -F  flip vertically, if the image is upside down\n"
		    "  -v  report every input event rather than a sample\n");
		return (1);
	}

	v.sock = connect_socket(argv[0]);
	if (v.sock < 0) {
		fprintf(stderr, "bhyve-viewer: cannot connect to %s: %s\n",
		    argv[0], strerror(errno));
		return (1);
	}

	v.dpy = wl_display_connect(NULL);
	if (v.dpy == NULL) {
		fprintf(stderr, "bhyve-viewer: no Wayland display "
		    "(is WAYLAND_DISPLAY set?)\n");
		return (1);
	}
	v.registry = wl_display_get_registry(v.dpy);
	wl_registry_add_listener(v.registry, &registry_listener, &v);
	wl_display_roundtrip(v.dpy);

	if (v.compositor == NULL || v.wm_base == NULL) {
		fprintf(stderr, "bhyve-viewer: compositor lacks "
		    "wl_compositor or xdg_wm_base\n");
		return (1);
	}
	if (!egl_setup(&v))
		return (1);

	v.surface = wl_compositor_create_surface(v.compositor);
	v.xsurface = xdg_wm_base_get_xdg_surface(v.wm_base, v.surface);
	xdg_surface_add_listener(v.xsurface, &xdg_surface_listener, &v);
	v.toplevel = xdg_surface_get_toplevel(v.xsurface);
	xdg_toplevel_add_listener(v.toplevel, &toplevel_listener, &v);
	xdg_toplevel_set_title(v.toplevel, "bhyve");
	wl_surface_commit(v.surface);
	wl_display_roundtrip(v.dpy);

	v.egl_window = wl_egl_window_create(v.surface, v.win_w, v.win_h);
	v.egl_surf = eglCreateWindowSurface(v.egl_dpy, v.egl_cfg,
	    (EGLNativeWindowType)v.egl_window, NULL);
	if (v.egl_surf == EGL_NO_SURFACE) {
		fprintf(stderr, "bhyve-viewer: eglCreateWindowSurface failed\n");
		return (1);
	}
	if (!eglMakeCurrent(v.egl_dpy, v.egl_surf, v.egl_surf, v.egl_ctx)) {
		fprintf(stderr, "bhyve-viewer: eglMakeCurrent failed\n");
		return (1);
	}
	if (!gl_setup(&v))
		return (1);
	/*
	 * Do not wait for the host compositor's frame callback inside
	 * eglSwapBuffers.  Blocking there ties this process's event loop to
	 * the host refresh while the guest presents independently, and any
	 * time spent blocked is time input is not dispatched.
	 */
	eglSwapInterval(v.egl_dpy, 0);

	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	signal(SIGHUP, on_signal);

	while (v.running && !quit_requested) {
		/*
		 * eglSwapBuffers() dispatches its own Wayland queue, and doing
		 * so drains the display fd.  Events for our proxies land in the
		 * default queue already read from the socket, so a later poll()
		 * reports nothing readable and they are never dispatched --
		 * which is why input arrived during startup and then stopped
		 * the moment frames began.
		 *
		 * prepare_read/read_events is the pattern that survives another
		 * party reading the same fd: anything already queued is
		 * dispatched before blocking, and the read is cancelled if we
		 * wake for a different reason.
		 */
		while (wl_display_prepare_read(v.dpy) != 0)
			wl_display_dispatch_pending(v.dpy);
		wl_display_flush(v.dpy);

		pfd[0].fd = wl_display_get_fd(v.dpy);
		pfd[0].events = POLLIN;
		pfd[1].fd = v.sock;
		pfd[1].events = POLLIN;

		if (poll(pfd, 2, -1) < 0) {
			wl_display_cancel_read(v.dpy);
			if (errno == EINTR)
				continue;	/* signal: loop re-checks quit */
			break;
		}

		if (pfd[0].revents & POLLIN)
			wl_display_read_events(v.dpy);
		else
			wl_display_cancel_read(v.dpy);
		if (wl_display_dispatch_pending(v.dpy) < 0)
			break;

		if ((pfd[1].revents & (POLLIN | POLLHUP)) &&
		    !sock_readable(&v)) {
			fprintf(stderr, "bhyve-viewer: bhyve closed the "
			    "connection\n");
			break;
		}

		if (v.have_pending) {
			struct buf *b = buf_find(&v, v.pending_buf);
			int fence = v.pending_fence;

			v.have_pending = false;
			v.pending_fence = -1;
			if (b != NULL && b->tex != 0)
				draw(&v, b, fence);
			else if (fence >= 0)
				close(fence);
		}
	}

	fprintf(stderr, "bhyve-viewer: sent %ju key and %ju pointer events\n",
	    (uintmax_t)keys_sent, (uintmax_t)ptrs_sent);

	/* Do not leave the guest holding keys because we exited. */
	for (uint32_t code = 0; code < 256; code++)
		if (key_is_down[code / 8] & (1u << (code % 8)))
			send_key(&v, code, false);

	eglMakeCurrent(v.egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE,
	    EGL_NO_CONTEXT);
	wl_display_disconnect(v.dpy);
	close(v.sock);
	return (0);
}
