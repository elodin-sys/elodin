/*
 ISC License

 Copyright (c) 2020, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef albedoSimMsg_H
#define albedoSimMsg_H

/*! albedo message definition */
typedef struct {
    // Maximum albedo acting on the instrument
    // considering the instrument's position 
    double albedoAtInstrumentMax;   //!< [-] Max albedo flux ratio at instrument
    double AfluxAtInstrumentMax;    //!< [W/m^2] Max albedo flux at instrument
    // Albedo acting on the instrument
    // considering the unit normal and fov of the instrument in addition to the position
    double albedoAtInstrument;      //!< [-] Albedo flux ratio at instrument
    double AfluxAtInstrument;       //!< [W/m^2] Albedo flux at instrument
}AlbedoMsgPayload;
#endif /* albedoSimMsg_h */
