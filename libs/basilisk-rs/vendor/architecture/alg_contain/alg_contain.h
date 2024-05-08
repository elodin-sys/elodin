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


#ifndef ALG_CONTAIN_H
#define ALG_CONTAIN_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"


typedef void (*AlgPtr)(void*, uint64_t);
typedef void (*AlgUpdatePtr)(void*, uint64_t, uint64_t);

/*! @brief AlgContain Class */
class AlgContain: public SysModel {
public:
    AlgContain();
    ~AlgContain();
    AlgContain(void *DataIn, void(*UpPtr) (void*, uint64_t, uint64_t),
        void (*SelfPtr)(void*, uint64_t)=NULL,
		void(*ResetPtr)(void*, uint64_t, uint64_t) = NULL); //!< constructor
    
    void UseData(void *IncomingData) {DataPtr = IncomingData;} //!< method
    void UseUpdate(void (*LocPtr)(void*, uint64_t, uint64_t)) {AlgUpdate = LocPtr;} //!< method
    void UseSelfInit(void (*LocPtr)(void*, uint64_t)) {AlgSelfInit = LocPtr;} //!< method
	void UseReset(void(*LocPtr)(void*, uint64_t, uint64_t)) { AlgReset = LocPtr; } //!< method
    void SelfInit(); //!< method
    void UpdateState(uint64_t CurrentSimNanos); //!< method
	void Reset(uint64_t CurrentSimNanos); //!< method
    uint64_t getSelfInitAddress() {return reinterpret_cast<uint64_t>(*AlgSelfInit);} //!< method
    uint64_t getResetAddress() {return reinterpret_cast<uint64_t>(*AlgReset);} //!< method
    uint64_t getUpdateAddress() {return reinterpret_cast<uint64_t>(*AlgUpdate);} //!< method
    
public:
    void *DataPtr;                              //!< class variable
    AlgPtr AlgSelfInit;                         //!< class variable
	AlgUpdatePtr AlgReset;                      //!< class variable
    AlgUpdatePtr AlgUpdate;                     //!< class variable
    BSKLogger bskLogger;                      //!< -- BSK Logging
};


#endif
