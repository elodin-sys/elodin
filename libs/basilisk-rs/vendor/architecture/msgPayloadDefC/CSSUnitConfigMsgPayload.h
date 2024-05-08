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

#ifndef CSS_UNIT_MESSAGE_H
#define CSS_UNIT_MESSAGE_H




/*! @brief Structure used to contain the configuration information for
 each sun sensor*/
typedef struct {
    double nHat_B[3];          //!< [-] CSS unit normal expressed in structure
    double CBias;              //!< [W]  Individual calibration coefficient bias for CSS.  If all CSS have the same gain, then this is set to 1.0. If one CSS has a 10% stronger response for the same input, then the value would be 1.10 
}CSSUnitConfigMsgPayload;


#endif
