/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef MTB_ARRAY_CONFIG_MSG_H
#define MTB_ARRAY_CONFIG_MSG_H

#include "architecture/utilities/macroDefinitions.h"

/*! @brief magnetic torque bar array configuration msg */
typedef struct{
    int    numMTB;                      //!< [-] number of magnetic torque bars on the spacecraft
    double GtMatrix_B[3*MAX_EFF_CNT];   //!< [-] magnetic torque bar alignment matrix in Body frame components, must be provided in row-major format
    double maxMtbDipoles[MAX_EFF_CNT];  //!< [A-m2] maximum commandable dipole for each magnetic torque bar
}MTBArrayConfigMsgPayload;


#endif

