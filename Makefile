# bhyve-view: present a bhyve guest's GPU output in a host Wayland window.
#
# Plain BSD make; the only dependencies are wayland-client and the protocol
# XML from wayland-protocols, both found through pkg-config.

PROG=	bhyve-view
SRCS=	main.c xdg-shell-protocol.c linux-dmabuf-v1-protocol.c
MAN=

BINDIR?=	/usr/local/bin

WAYLAND_PROTOCOLS!=	pkg-config --variable=pkgdatadir wayland-protocols
WAYLAND_SCANNER!=	pkg-config --variable=wayland_scanner wayland-scanner

CFLAGS+=	-I${.CURDIR}/src -I${.OBJDIR} -Wall -Wextra
CFLAGS+=	`pkg-config --cflags wayland-client`
LDADD+=		`pkg-config --libs wayland-client`

.PATH: ${.CURDIR}/src

# Protocol glue is generated, not vendored: it must match the installed
# wayland-protocols rather than a copy that silently goes stale.
xdg-shell-client-protocol.h:
	${WAYLAND_SCANNER} client-header \
	    ${WAYLAND_PROTOCOLS}/stable/xdg-shell/xdg-shell.xml $@

xdg-shell-protocol.c:
	${WAYLAND_SCANNER} private-code \
	    ${WAYLAND_PROTOCOLS}/stable/xdg-shell/xdg-shell.xml $@

linux-dmabuf-v1-client-protocol.h:
	${WAYLAND_SCANNER} client-header \
	    ${WAYLAND_PROTOCOLS}/stable/linux-dmabuf/linux-dmabuf-v1.xml $@

linux-dmabuf-v1-protocol.c:
	${WAYLAND_SCANNER} private-code \
	    ${WAYLAND_PROTOCOLS}/stable/linux-dmabuf/linux-dmabuf-v1.xml $@

main.o: xdg-shell-client-protocol.h linux-dmabuf-v1-client-protocol.h

CLEANFILES+=	xdg-shell-client-protocol.h xdg-shell-protocol.c \
		linux-dmabuf-v1-client-protocol.h linux-dmabuf-v1-protocol.c

.include <bsd.prog.mk>
