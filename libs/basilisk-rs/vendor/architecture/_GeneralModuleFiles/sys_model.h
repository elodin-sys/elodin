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

#ifndef _SysModel_HH_
#define _SysModel_HH_

#include <string>
#include <stdint.h>
#include <architecture/utilities/bskLogging.h>

/*! @brief Simulation System Model Class */
class SysModel
{
public:
    SysModel();
    SysModel(const SysModel &obj); //!< constructor definition

    virtual ~SysModel(){};

    /** Initializes the module, create messages */
    virtual void SelfInit(){};

    /** ??? */
    virtual void IntegratedInit(){};

    /** Reads incoming messages, performs module actions, writes output messages */
    virtual void UpdateState(uint64_t CurrentSimNanos){};

    /** Called at simulation initialization, resets module to specified time */
    virtual void Reset(uint64_t CurrentSimNanos){};

public:
    std::string ModelTag = "";     //!< -- name for the algorithm to base off of
    uint64_t CallCounts = 0;       //!< -- Counts on the model being called
    uint32_t RNGSeed = 0x1badcad1; //!< -- Giving everyone a random seed for ease of MC
    int64_t moduleID;              //!< -- Module ID for this module  (handed out by module_id_generator)
};

// The following code helps users who defined their own module classes
// to transition to using the SWIG file for sys_model instead of the header file.
// After a period of 12 months from 2023/09/15, this message can be removed.
#ifdef SWIG
%extend SysModel
{
    %pythoncode %{
        def logger(self, *args, **kwargs):
            raise TypeError(
                f"The 'logger' function is not supported for this type ('{type(self).__qualname__}'). "
                "To fix this, update the SWIG file for this module. Change "
                """'%include "sys_model.h"' to '%include "sys_model.i"'"""
            )
    %}
}
#endif

#endif /* _SYS_MODEL_H_ */
