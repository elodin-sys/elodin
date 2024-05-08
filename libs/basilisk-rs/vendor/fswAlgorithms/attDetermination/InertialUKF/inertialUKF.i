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
%module inertialUKF
%{
   #include "inertialUKF.h"
   #include "architecture/utilities/ukfUtilities.h"
%}

%include "swig_conly_data.i"
STRUCTASLIST(LowPassFilterData)
STRUCTASLIST(STMessage)

%include "swig_c_wrap.i"
%c_wrap_2(inertialUKF, InertialUKFConfig);


%include "architecture/utilities/ukfUtilities.h"

%include "architecture/msgPayloadDefC/InertialFilterMsgPayload.h"
struct InertialFilterMsg_C;
%include "architecture/msgPayloadDefC/STAttMsgPayload.h"
struct STAttMsg_C;
%include "architecture/msgPayloadDefC/VehicleConfigMsgPayload.h"
struct VehicleConfigMsg_C;
%include "architecture/msgPayloadDefC/RWArrayConfigMsgPayload.h"
struct RWArrayConfigMsg_C;
%include "architecture/msgPayloadDefC/RWSpeedMsgPayload.h"
struct RWSpeedMsg_C;
%include "architecture/msgPayloadDefC/AccDataMsgPayload.h"
struct AccDataMsg_C;
%include "architecture/msgPayloadDefC/AccPktDataMsgPayload.h"
struct AccPktDataMsg_C;
%include "architecture/msgPayloadDefC/NavAttMsgPayload.h"
struct NavAttMsg_C;

%include "architecture/utilities/signalCondition.h"
struct LowPassFilterData;

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}

