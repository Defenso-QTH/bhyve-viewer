/*-
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Copyright (c) 2026 Defenso
 */

/*
 * evdev keycode -> XT set-1 scancode.
 *
 * Wayland delivers evdev keycodes.  bhyve's ps2kbd wants an XT set-1 scancode:
 * ps2kbd_keysym_queue() indexes keyset1to2_translations[keycode & 0x7f] and
 * treats bit 0x80 as "prefix this with E0".  Passing an evdev code straight
 * through would land on the wrong key for anything above the main block.
 *
 * For evdev 1..83 the two happen to coincide -- the evdev numbering was taken
 * from the XT set, so KEY_ESC is 1 is 0x01 and so on up to KEY_KPDOT.  Beyond
 * that they diverge and the extended keys need the E0 flag, so the table is
 * explicit rather than clever.
 *
 * 0 means "no mapping"; the caller should then fall back to sending a keysym
 * and letting bhyve's ASCII table deal with it.
 */

#ifndef _BHYVE_VIEW_KEYMAP_H_
#define	_BHYVE_VIEW_KEYMAP_H_

#include <stdint.h>

#define	XT_E0	0x80	/* bhyve reads this bit as "E0 prefix" */

static inline uint32_t
evdev_to_xt(uint32_t evdev)
{
	/* Main block: identical numbering, no prefix. */
	if (evdev >= 1 && evdev <= 83)
		return (evdev);

	switch (evdev) {
	case 87:  return (0x57);		/* F11 */
	case 88:  return (0x58);		/* F12 */

	/* Keypad and right-hand modifiers live behind E0. */
	case 96:  return (0x1c | XT_E0);	/* KP Enter */
	case 97:  return (0x1d | XT_E0);	/* Right Ctrl */
	case 98:  return (0x35 | XT_E0);	/* KP Slash */
	case 100: return (0x38 | XT_E0);	/* Right Alt */

	case 102: return (0x47 | XT_E0);	/* Home */
	case 103: return (0x48 | XT_E0);	/* Up */
	case 104: return (0x49 | XT_E0);	/* Page Up */
	case 105: return (0x4b | XT_E0);	/* Left */
	case 106: return (0x4d | XT_E0);	/* Right */
	case 107: return (0x4f | XT_E0);	/* End */
	case 108: return (0x50 | XT_E0);	/* Down */
	case 109: return (0x51 | XT_E0);	/* Page Down */
	case 110: return (0x52 | XT_E0);	/* Insert */
	case 111: return (0x53 | XT_E0);	/* Delete */

	case 125: return (0x5b | XT_E0);	/* Left Meta */
	case 126: return (0x5c | XT_E0);	/* Right Meta */
	case 127: return (0x5d | XT_E0);	/* Menu */

	default:  return (0);
	}
}

#endif /* _BHYVE_VIEW_KEYMAP_H_ */
