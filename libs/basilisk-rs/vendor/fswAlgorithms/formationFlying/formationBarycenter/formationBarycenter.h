/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#ifndef FORMATION_BARYCENTER_H
#define FORMATION_BARYCENTER_H

#include <vector>

#include "architecture/msgPayloadDefC/VehicleConfigMsgPayload.h"
#include "cMsgCInterface/NavTransMsg_C.h"

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"

/*! @brief This module computes the barycenter of a swarm of satellites, either using cartesian coordinates or orbital elements. 
 */
class FormationBarycenter: public SysModel {
public:
    FormationBarycenter();
    ~FormationBarycenter();

    void SelfInit();
    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);
    void ReadInputMessages();
    void addSpacecraftToModel(Message<NavTransMsgPayload>* tmpScNavMsg, Message<VehicleConfigMsgPayload>* tmpScPayloadMsg);
    void computeBaricenter();
    void WriteOutputMessage(uint64_t CurrentClock);

public:
    std::vector<ReadFunctor<NavTransMsgPayload>> scNavInMsgs;  //!< spacecraft navigation input msg
    std::vector<ReadFunctor<VehicleConfigMsgPayload>> scPayloadInMsgs;  //!< spacecraft payload input msg

    Message<NavTransMsgPayload> transOutMsg;    //!< translation navigation output msg   
    NavTransMsg_C transOutMsgC = {};        //!< C-wrapped translation navigation output msg, zeroed

    bool useOrbitalElements;        //!< flag that determines whether to use cartesian or orbital elementd weighted averaging
    double mu;      //!< gravitational parameter to be used with orbital elements averaging

    BSKLogger bskLogger;              //!< -- BSK Logging

private:
    std::vector<NavTransMsgPayload> scNavBuffer;             //!< buffer of spacecraft navigation info
    std::vector<VehicleConfigMsgPayload> scPayloadBuffer;             //!< buffer of spacecraft payload

    NavTransMsgPayload transOutBuffer;      //!< buffer for the output message

};

#endif
