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

#ifndef bodyHeadingSimMsg_h
#define bodyHeadingSimMsg_h


//!@brief Planet heading message definition.
/*! Many modules in Basilisk utilize the spacecraft body frame heading to something.
  For instance, a spacecraft may want to point at a point on earth, the sun, another planet
  or another spacecraft. This message is unique from the interface message NavAttMsgPayload
  in being agnostic of the thing being pointed to while not including separate information
  about attitude and rates that are not necessarily desired
 */
typedef struct {
    double rHat_XB_B[3];  //!< [] unit heading vector to any thing "X" in the spacecraft, "B", body frame
}BodyHeadingMsgPayload;


#endif /* bodyHeadingSimMsg_h */
