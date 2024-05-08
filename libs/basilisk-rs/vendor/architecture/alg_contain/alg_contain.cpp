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
#include "alg_contain.h"

AlgContain::AlgContain()
{
    AlgSelfInit = NULL;
    AlgUpdate = NULL;
    DataPtr = NULL;
	AlgReset = NULL;
    CallCounts = 0;
    return;
}

AlgContain::AlgContain(void *DataIn, void(*UpPtr) (void*, uint64_t, uint64_t),
                       void (*SelfPtr)(void*, uint64_t),
	                   void (*ResetPtr) (void*, uint64_t, uint64_t))
{
    DataPtr = DataIn;
    AlgSelfInit = SelfPtr;
    AlgUpdate = UpPtr;
	AlgReset = ResetPtr;
}

AlgContain::~AlgContain()
{
    return;
}

void AlgContain::SelfInit()
{
    if(AlgSelfInit != NULL)
    {
        AlgSelfInit(DataPtr, (uint32_t) moduleID);
    }
}


void AlgContain::UpdateState(uint64_t CurrentSimNanos)
{
    if(AlgUpdate != NULL)
    {
        AlgUpdate(DataPtr, CurrentSimNanos, (uint32_t) moduleID);
    }
}

void AlgContain::Reset(uint64_t CurrentSimNanos)
{
	if (AlgReset != NULL)
	{
		AlgReset(DataPtr, CurrentSimNanos, (uint32_t) moduleID);
	}
}
