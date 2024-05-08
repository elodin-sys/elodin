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

#ifndef _ModuleIdGenerator_HH_
#define _ModuleIdGenerator_HH_

#include <inttypes.h>

/*! @brief module ID generating class */
#ifdef _WIN32
class __declspec( dllexport) ModuleIdGenerator
#else
class ModuleIdGenerator
#endif
{
public:
    int64_t checkoutModuleID();  //! -- Assigns next integer module ID
    static ModuleIdGenerator* GetInstance();  //! -- returns a pointer to the sim instance of ModuleIdGenerator

private:
    int64_t nextModuleID;  //!< the next module ID to give out when a module (SysModel sub-class) comes online
    static ModuleIdGenerator *TheInstance;        //!< instance of simulation module

    ModuleIdGenerator();
    ~ModuleIdGenerator();
    ModuleIdGenerator(ModuleIdGenerator const &) {};
    ModuleIdGenerator& operator =(ModuleIdGenerator const &){return(*this);};
    
};

#endif /* _ModuleIdGenerator_HH_ */
