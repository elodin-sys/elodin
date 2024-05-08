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
%module rwMotorVoltage
%{
   #include "rwMotorVoltage.h"
%}

%include "swig_c_wrap.i"
%c_wrap(rwMotorVoltage);

%include "architecture/msgPayloadDefC/ArrayMotorTorqueMsgPayload.h"
struct ArrayMotorTorqueMsg_C;
%include "architecture/msgPayloadDefC/RWAvailabilityMsgPayload.h"
struct RWAvailabilityMsg_C;
%include "architecture/msgPayloadDefC/RWArrayConfigMsgPayload.h"
struct RWArrayConfigMsg_C;
%include "architecture/msgPayloadDefC/RWSpeedMsgPayload.h"
struct RWSpeedMsg_C;
%include "architecture/msgPayloadDefC/ArrayMotorVoltageMsgPayload.h"
struct ArrayMotorVoltageMsg_C;

%include "fswAlgorithms/fswUtilities/fswDefinitions.h"
%include "architecture/utilities/macroDefinitions.h"

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}
