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
%module bskUtilities
%{
   #include "architecture/utilities/macroDefinitions.h"
   #include "fswAlgorithms/fswUtilities/fswDefinitions.h"
   #include "simulation/dynamics/reactionWheels/reactionWheelSupport.h"
   #include <vector>
%}

%include "std_vector.i"
%include "std_string.i"
%include "swig_eigen.i"
%include "swig_conly_data.i"
%include "stdint.i"

%include "architecture/utilities/macroDefinitions.h"
%include "fswAlgorithms/fswUtilities/fswDefinitions.h"
%include "simulation/dynamics/reactionWheels/reactionWheelSupport.h"

%template(Eigen3dVector) std::vector<Eigen::Vector3d, std::allocator<Eigen::Vector3d>>;

%pythoncode %{
import sys
protectAllClasses(sys.modules[__name__])
%}
