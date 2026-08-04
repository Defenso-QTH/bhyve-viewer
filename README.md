# bhyve-view

Presents a bhyve guest's GPU output in a window on the host's Wayland
compositor, with no copy and no video encoding anywhere in the path.

## Why

A guest using virtio-gpu with venus renders on the *host* GPU: its images live
in host GPU memory the whole time.  The usual arrangement — a compositor and a
VNC server inside the guest — therefore ships the framebuffer *into* the guest
so the guest can capture it and encode it back out again.  Measured on a real
workload, that cost four guest vCPUs pegged at ~96% while the host renderer sat
at 3.4%, and it was insensitive to resolution: the cost was capture and encode,
not drawing.

Instead, bhyve exports the scanout as a dma_buf and passes the file descriptor
over a unix socket.  This program imports it with `zwp_linux_dmabuf_v1` and
hands it to the host compositor.  The guest does no display work at all.

It is a separate process rather than code inside bhyve so that the VMM keeps no
Wayland or EGL dependencies, and so this can be jailed with nothing but the
socket.

## Requires

- bhyve with the virtio-gpu device built `BHYVE_VIRGL_SUPPORT=yes` and started
  with `display=unix:/path/to/socket`
- a host compositor offering `zwp_linux_dmabuf_v1`
- the guest scanout must be exportable — bhyve logs
  `dmabuf_export=YES` when it is

## Use

    bhyve-view /path/to/socket

Keyboard and pointer events are forwarded to the guest through bhyve's existing
console layer, so they arrive at whatever keyboard and tablet the guest was
configured with.

## Status

Early.  The dma_buf import and fd passing are the parts most likely to need
iteration.  Only the DMABUF transport is implemented; the shm fallback that
bhyve can fall back to is not handled yet.
