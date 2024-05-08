/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#ifndef SIM_FSW_MACROS_H
#define SIM_FSW_MACROS_H

#define MAX_CIRCLE_NUM 10
#define MAX_LIMB_PNTS 2000
#define MAX_EFF_CNT 36
#define MAX_NUM_CSS_SENSORS 32
#define MAX_ST_VEH_COUNT 4

#define NANO2SEC        1e-9
#define SEC2NANO        1e9
#define RECAST6X6       (double (*)[6])
#define RECAST3X3       (double (*)[3])
#define RECAST2x2       (double (*)[2])
#define SEC2HOUR        1./3600.



#endif
