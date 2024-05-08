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

#include "sim_model.h"
#include <cstring>
#include <iostream>

void activateNewThread(void *threadData)
{

    SimThreadExecution *theThread = static_cast<SimThreadExecution*> (threadData);

    //std::cout << "Starting thread yes" << std::endl;
    theThread->postInit();

    while(theThread->threadValid())
    {
        theThread->lockThread();
        if(theThread->selfInitNow){
            theThread->selfInitProcesses();
            theThread->selfInitNow = false;
        }
        else if(theThread->crossInitNow){
            theThread->crossInitProcesses();
            theThread->crossInitNow = false;
        }
        else if(theThread->resetNow){
            theThread->resetProcesses();
            theThread->resetNow = false;
        }
        else{
            theThread->StepUntilStop();
        }
        //std::cout << "Stepping thread"<<std::endl;
        theThread->unlockParent();

    }
    //std::cout << "Killing thread" << std::endl;

}

SimThreadExecution::SimThreadExecution(uint64_t threadIdent, uint64_t currentSimNanos) : SimThreadExecution(){

    currentThreadNanos = currentSimNanos;
    threadID = threadIdent;
}

SimThreadExecution::~SimThreadExecution() {

}

SimThreadExecution::SimThreadExecution() {
    currentThreadNanos = 0;
    threadRunning = false;
    terminateThread = false;
    selfInitNow = false;
    crossInitNow = false;
    resetNow = false;
    threadID = 0;
    CurrentNanos = 0;
    NextTaskTime = 0;
    stopThreadNanos=0;
    nextProcPriority = -1;
    threadContext = nullptr;

}

/*! This method provides a synchronization mechanism for the "child" thread
    ensuring that it can be held at a fixed point after it finishes the
    execution of a given frame until it is released by the "parent" thread.
 @return void
 */
void SimThreadExecution::lockThread() {
    this->selfThreadLock.acquire();
}

/*! This method provides a forced synchronization on the "parent" thread so that
    the parent and all other threads in the system can be forced to wait at a
    known time until this thread has finished its execution for that time.
 @return void
 */
void SimThreadExecution::lockParent() {
    this->parentThreadLock.acquire();
}

/*! This method provides an entry point for the "parent" thread to release the
    child thread for a single frame's execution.  It is intended to only be
    called from the parent thread.
 @return void
 */
void SimThreadExecution::unlockThread() {
    this->selfThreadLock.release();
}

/*! This method provides an entry point for the "child" thread to unlock the
    parent thread after it has finished its execution in a frame.  That way the
    parent and all of its other children have to wait for this child to finish
    its execution.
 @return void
 */
void SimThreadExecution::unlockParent() {
    this->parentThreadLock.release();
}

/*! This method steps all of the processes forward to the current time.  It also
    increments the internal simulation time appropriately as the simulation
    processes are triggered
    @param stopPri The priority level below which the sim won't go
    @return void
*/
void SimThreadExecution::SingleStepProcesses(int64_t stopPri)
{
    uint64_t nextCallTime = ~((uint64_t) 0);
    std::vector<SysProcess *>::iterator it = this->processList.begin();
    this->CurrentNanos = this->NextTaskTime;
    while(it!= this->processList.end() && this->threadValid())
    {
        SysProcess *localProc = (*it);
        if(localProc->processEnabled())
        {
            while(localProc->nextTaskTime < this->CurrentNanos ||
                  (localProc->nextTaskTime == this->CurrentNanos &&
                   localProc->processPriority >= stopPri))
            {
                localProc->singleStepNextTask(this->CurrentNanos);
            }
            if(localProc->getNextTime() < nextCallTime)
            {
                nextCallTime = localProc->getNextTime();
                this->nextProcPriority = localProc->processPriority;
            }
            else if(localProc->getNextTime() == nextCallTime &&
                    localProc->processPriority > this->nextProcPriority)
            {
                this->nextProcPriority = localProc->processPriority;
            }
        }
        it++;
    }
    this->NextTaskTime = nextCallTime != ~((uint64_t) 0) ? nextCallTime : this->CurrentNanos;

}

/*! This method steps the simulation until the specified stop time and
 stop priority have been reached.
 @return void
 */
void SimThreadExecution::StepUntilStop()
{
    /*! - Note that we have to step until both the time is greater and the next
     Task's start time is in the future. If the NextTaskTime is less than
     SimStopTime, then the inPri shouldn't come into effect, so set it to -1
     (that's less than all process priorities, so it will run through the next
     process)*/
    int64_t inPri = stopThreadNanos == this->NextTaskTime ? stopThreadPriority : -1;
    while(this->threadValid() && (this->NextTaskTime < stopThreadNanos || (this->NextTaskTime == stopThreadNanos &&
                                               this->nextProcPriority >= stopThreadPriority)) )
    {
        this->SingleStepProcesses(inPri);
        inPri = stopThreadNanos == this->NextTaskTime ? stopThreadPriority : -1;
    }
}

/*! This method is currently vestigial and needs to be populated once the message
    sharing process between different threads is handled.
    TODO: Make this method move messages safely between threads
 @return void
 */
void SimThreadExecution::moveProcessMessages() {
    std::vector<SysProcess *>::iterator it;
    for(it = this->processList.begin(); it != this->processList.end(); it++)
    {
        //(*it)->routeInterfaces(this->CurrentNanos);
    }

}

/*! Once threads are released for execution, this method ensures that they finish
    their startup before the system starts to go through its initialization
    activities.  It's very similar to the locking process, but provides different
    functionality.
 @return void
 */
void SimThreadExecution::waitOnInit() {
    std::unique_lock<std::mutex> lck(this->initReadyLock);
    while(!this->threadActive())
    {
        (this)->initHoldVar.wait(lck);
    }
}

/*! This method allows the startup activities to alert the parent thread once
    they have cleared their construction phase and are ready to go through
    initialization.
 @return void
 */
void SimThreadExecution::postInit() {
    std::unique_lock<std::mutex> lck(this->initReadyLock);
    this->threadReady();
    this->initHoldVar.notify_one();
}

/*! This method is used by the "child" thread to walk through all of its tasks
    and processes and initialize them serially.  Note that other threads can also
    be initializing their systems simultaneously.
 @return void
 */
void SimThreadExecution::selfInitProcesses() {
    std::vector<SysProcess *>::iterator it;
    for(it=this->processList.begin(); it!= this->processList.end(); it++)
    {
        (*it)->selfInitProcess();
    }
}

/*! This method is vestigial and should probably be removed once MT message
    movement has been completed.
 @return void
 */
void SimThreadExecution::crossInitProcesses() {
    std::vector<SysProcess *>::iterator it;
    for(it=this->processList.begin(); it!= this->processList.end(); it++)
    {
        //(*it)->crossInitProcess();
    }
}

/*! This method allows the "child" thread to reset both its timing/scheduling, as
    well as all of its allocated tasks/modules when commanded.  This is always
    called during init, but can be called during runtime as well.
 @return void
 */
void SimThreadExecution::resetProcesses() {
    std::vector<SysProcess *>::iterator it;
    this->currentThreadNanos = 0;
    this->CurrentNanos = 0;
    this->NextTaskTime = 0;
    for(it=this->processList.begin(); it!= this->processList.end(); it++)
    {
        (*it)->resetProcess(this->currentThreadNanos);
    }
}

/*! This method pops a new process onto the execution stack for the "child"
    thread.  It allows the user to put specific processes onto specific threads
    if that is desired.
 @return void
 */
void SimThreadExecution::addNewProcess(SysProcess* newProc) {
    processList.push_back(newProc);
    newProc->setProcessControlStatus(true);
}

/*! This Constructor is used to initialize the top-level sim model.
 */
SimModel::SimModel()
{

    this->threadList.clear();

    //Default to single-threaded runtime
    SimThreadExecution *newThread = new SimThreadExecution(0, 0);
    this->threadList.push_back(newThread);

    this->NextTaskTime = 0;

    this->CurrentNanos = 0;
    this->NextTaskTime = 0;
    this->nextProcPriority = -1;
}

/*! Nothing to destroy really */
SimModel::~SimModel()
{
    this->deleteThreads();
}

/*! This method steps the simulation until the specified stop time and
 stop priority have been reached.
 @param SimStopTime Nanoseconds to step the simulation for
 @param stopPri The priority level below which the sim won't go
 @return void
 */
void SimModel::StepUntilStop(uint64_t SimStopTime, int64_t stopPri)
{
    std::vector<SimThreadExecution*>::iterator thrIt;
    std::cout << std::flush;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->moveProcessMessages();
    }
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->stopThreadNanos = SimStopTime;
        (*thrIt)->stopThreadPriority = stopPri;
        if((*thrIt)->procCount() > 0) {
            (*thrIt)->unlockThread();
        }
    }
    this->NextTaskTime = (uint64_t) ~0;
    this->CurrentNanos = (uint64_t) ~0;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        if((*thrIt)->procCount() > 0) {
            (*thrIt)->lockParent();
            this->NextTaskTime = (*thrIt)->NextTaskTime < this->NextTaskTime ?
                                 (*thrIt)->NextTaskTime : this->NextTaskTime;
            this->CurrentNanos = (*thrIt)->CurrentNanos < this->CurrentNanos ?
                                 (*thrIt)->CurrentNanos : this->CurrentNanos;
        }
    }
}


/*! This method allows the user to attach a process to the simulation for
    execution.  Note that the priority level of the process determines what
    order it gets called in: higher priorities are called before lower
    priorities. If priorities are the same, the proc added first goes first.
    @return void
    @param newProc the new process to be added
*/
void SimModel::addNewProcess(SysProcess *newProc)
{
    std::vector<SysProcess *>::iterator it;
    for(it = this->processList.begin(); it != this->processList.end(); it++)
    {
        if(newProc->processPriority > (*it)->processPriority)
        {
            this->processList.insert(it, newProc);
            return;
        }
    }
    this->processList.push_back(newProc);
}

/*! This method goes through all of the processes in the simulation,
 *  all of the tasks within each process, and all of the models within
 *  each task and self-inits them.
 @return void
 */
void SimModel::selfInitSimulation()
{
    std::vector<SimThreadExecution*>::iterator thrIt;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->selfInitNow = true;
        (*thrIt)->unlockThread();
    }
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++) {
        (*thrIt)->lockParent();
    }
    this->NextTaskTime = 0;
    this->CurrentNanos = 0;

}

/*! This method goes through all of the processes in the simulation,
 *  all of the tasks within each process, and all of the models within
 *  each task and resets them.
 @return void
 */
void SimModel::resetInitSimulation()
{


    std::vector<SimThreadExecution*>::iterator thrIt;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->resetNow = true;
        (*thrIt)->unlockThread();
    }
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->lockParent();

    }
}

/*! This method steps all of the processes forward to the current time.  It also
    increments the internal simulation time appropriately as the simulation
    processes are triggered
    @param stopPri The priority level below which the sim won't go
    @return void
*/

void SimModel::SingleStepProcesses(int64_t stopPri)
{
    uint64_t nextCallTime = ~((uint64_t) 0);
    std::vector<SysProcess *>::iterator it = this->processList.begin();
    this->CurrentNanos = this->NextTaskTime;
    while(it!= this->processList.end())
    {
        SysProcess *localProc = (*it);
        if(localProc->processEnabled())
        {
            while(localProc->nextTaskTime < this->CurrentNanos ||
                (localProc->nextTaskTime == this->CurrentNanos &&
                  localProc->processPriority >= stopPri))
            {
                localProc->singleStepNextTask(this->CurrentNanos);
            }
            if(localProc->getNextTime() < nextCallTime)
            {
                nextCallTime = localProc->getNextTime();
                this->nextProcPriority = localProc->processPriority;
            }
            else if(localProc->getNextTime() == nextCallTime &&
                localProc->processPriority > this->nextProcPriority)
            {
                this->nextProcPriority = localProc->processPriority;
            }
        }
        it++;
    }

    this->NextTaskTime = nextCallTime != ~((uint64_t) 0) ? nextCallTime : this->CurrentNanos;
    //! - If a message has been added to logger, link the message IDs
}

/*! This method is used to reset a simulation to time 0. It sets all process and
 * tasks back to the initial call times. It clears all message logs. However,
 * it does not clear all message buffers and does not reset individual models.
 @return void
 */
void SimModel::ResetSimulation()
{
    std::vector<SysProcess *>::iterator it;
    //! - Iterate through model list and call the Task model initializer
    for(it = this->processList.begin(); it != this->processList.end(); it++)
    {
        (*it)->reInitProcess();
    }
    std::vector<SimThreadExecution*>::iterator thrIt;
    this->NextTaskTime = 0;
    this->CurrentNanos = 0;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->NextTaskTime = 0;
        (*thrIt)->CurrentNanos = 0;
    }
}

/*! This method removes all of the active processes from the "thread pool" that
    has been established.  It is needed during init and if sims are restarted or
    threads need to be reallocated.  Otherwise it is basically a no-op.
 @return void
 */
void SimModel::clearProcsFromThreads() {

    std::vector<SimThreadExecution*>::iterator thrIt;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->clearProcessList();
    }
    std::vector<SysProcess *>::iterator it;
    //! - Iterate through model list and call the Task model initializer
    for(it = this->processList.begin(); it != this->processList.end(); it++)
    {
        (*it)->setProcessControlStatus(false);
    }

}

/*! This method provides an easy mechanism for allowing the user to change the
    number of concurrent threads that will be executing in a given simulation.
    You tell the method how many threads you want in the system, it clears out
    any existing thread data, and then allocates fresh threads for the runtime.
 @param threadCount number of threads
 @return void
 */
void SimModel::resetThreads(uint64_t threadCount)
{

    this->clearProcsFromThreads();
    this->deleteThreads();
    this->threadList.clear();
    for(uint64_t i=0; i<threadCount; i++)
    {
        SimThreadExecution *newThread = new SimThreadExecution(0, 0);
        this->threadList.push_back(newThread);
    }

}

/*! This method walks through all of the child threads that have been created in
    the system, detaches them from the architecture, and then cleans up any
    memory that has been allocated to them in the architecture.  It just ensures
    clean shutdown of any existing runtime stuff.
 @return void
 */
void SimModel::deleteThreads() {
    std::vector<SimThreadExecution*>::iterator thrIt;
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->killThread();
        (*thrIt)->unlockThread();
        if((*thrIt)->threadContext && (*thrIt)->threadContext->joinable()) {
            (*thrIt)->threadContext->join();
            delete (*thrIt)->threadContext;
        }
        delete (*thrIt);
    }
    this->threadList.clear();
}

/*! This method provides a seamless allocation of processes onto active threads
    for any processes that haven't already been placed onto a thread.  If the
    user has allocated N threads, this method just walks through those threads
    and pops all of the processes onto those threads in a round-robin fashion.
 @return void
 */
void SimModel::assignRemainingProcs() {

    std::vector<SysProcess *>::iterator it;
    std::vector<SimThreadExecution*>::iterator thrIt;
    for(it=this->processList.begin(), thrIt=threadList.begin(); it!= this->processList.end(); it++, thrIt++)
    {
        if(thrIt == threadList.end())
        {
            thrIt = threadList.begin();
        }
        if((*it)->getProcessControlStatus()) {
            thrIt--; //Didn't get a thread to add, so roll back
        }
        else
        {
            (*thrIt)->addNewProcess((*it));
        }
    }
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        it=this->processList.begin();
        (*thrIt)->nextProcPriority = (*it)->processPriority;
        (*thrIt)->NextTaskTime = 0;
        (*thrIt)->CurrentNanos = 0;
        //(*thrIt)->lockThread();
        (*thrIt)->threadContext = new std::thread(activateNewThread, (*thrIt));
    }
    for(thrIt=this->threadList.begin(); thrIt != this->threadList.end(); thrIt++)
    {
        (*thrIt)->waitOnInit();
    }
}

/*! This method allows the user to specifically place a given process onto a
    specific thread index based on the currently active thread-pool.  This is the
    mechanism that a user has to specifically spread out processing in a way that
    makes the best sense to them.  Otherwise it happens in the round-robin
    manner described in the allocate-remaining-processes method.
 @param newProc The process that needs to get emplaced on the specified thread
 @param threadSel The thread index in the thread-pool that the process gets added
                  to
 @return void
 */
void SimModel::addProcessToThread(SysProcess *newProc, uint64_t threadSel)
{
    std::vector<SimThreadExecution*>::iterator thrIt;
    thrIt=threadList.begin() + threadSel;
    (*thrIt)->addNewProcess(newProc);
}



