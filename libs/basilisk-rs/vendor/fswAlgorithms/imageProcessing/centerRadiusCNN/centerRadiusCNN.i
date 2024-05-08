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
%module centerRadiusCNN
%{
   #include "centerRadiusCNN.h"
%}

%pythoncode %{
from Basilisk.architecture.swig_common_model import *
%}
%include "stdint.i"
%include "std_string.i"
%include "swig_conly_data.i"
%include "sys_model.i"
%include "centerRadiusCNN.h"

%include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
struct OpNavCirclesMsg_C;
%include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
struct CameraImageMsg_C;

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}
