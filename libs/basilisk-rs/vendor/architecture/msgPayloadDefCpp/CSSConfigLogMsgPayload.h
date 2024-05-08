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

#ifndef CSSCONFIGLOGSIMMSG_H
#define CSSCONFIGLOGSIMMSG_H


//!@brief CSS configuration message log message
/*! This message is the outpout of each CSS device to log all the configuration and
    measurement states.
 */
typedef struct
//@cond DOXYGEN_IGNORE
CSSConfigLogMsgPayload
//@endcond
{
    double r_B[3] = {0};    //!< [m] sensor position vector in the spacecraft, "B", body frame
    double nHat_B[3];       //!< [] sensor unit direction vector in the spacecraft, "B", body frame
    double fov;             //!< [rad] field of view (boresight to edge)
    double signal;          //!< [] current sensor signal
    double maxSignal;       //!< [] maximum sensor signal, -1 means this value is not set
    double minSignal;       //!< [] maximum sensor signal, -1 means this value is not set
    int    CSSGroupID = 0;  //!< [] Group ID if the CSS is part of a cluster
}CSSConfigLogMsgPayload;


#endif /* CSSCONFIGLOGSIMMSG_H */
