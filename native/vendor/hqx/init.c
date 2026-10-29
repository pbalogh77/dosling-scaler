/* Dosling modification notice, 2026-10-05:
 * Original changes made 2026-09-21: signed chroma conversion in init.c and
 * explicit signed differences in common.h avoid undefined conversions.
 * This is a modified HQx library; see ORIGIN.txt for provenance.
 */
/*
 * Copyright (C) 2010 Cameron Zemek ( grom@zeminvaders.net)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include <stdint.h>
#include "hqx.h"

/* Native scaler modification, 2026-10-17: the integer conversion in
 * common.h requires no global lookup or initialization. Keep this symbol
 * for the upstream HQx calling convention. */
HQX_API void HQX_CALLCONV hqxInit(void) {}
