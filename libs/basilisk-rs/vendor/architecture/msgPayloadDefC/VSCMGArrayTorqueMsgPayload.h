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

#ifndef FSW_VSCMG_TORQUE_H
#define FSW_VSCMG_TORQUE_H

#include "architecture/utilities/macroDefinitions.h"


/*! @brief Structure used to define the output definition for vehicle effectors*/
typedef struct {
    double wheelTorque[MAX_EFF_CNT];	//!< [N-m] VSCMG wheel torque array
	double gimbalTorque[MAX_EFF_CNT];	//!< [N-m] VSCMG gimbal torque array
}VSCMGArrayTorqueMsgPayload;





#endif
