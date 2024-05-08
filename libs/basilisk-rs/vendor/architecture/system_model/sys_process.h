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

#ifndef _SysProcess_HH_
#define _SysProcess_HH_

#include <vector>
#include <stdint.h>
#include "architecture/system_model/sys_model_task.h"
#include "architecture/utilities/bskLogging.h"

//! Structure that contains the information needed to call a Task
typedef struct {
    uint64_t NextTaskStart;  /*!< Time to call Task next*/
    uint64_t TaskUpdatePeriod;  /*!< Period of update for Task*/
    int32_t taskPriority;  /*!< [-] Priority level for the task*/
    SysModelTask *TaskPtr;  /*!< Handle to the Task that needs to be called*/
}ModelScheduleEntry;

//! Class used to group a set of tasks into one process (task group) of execution
class SysProcess
{
    
public:
    SysProcess();
    SysProcess(std::string messageContainer); //!< class method
    ~SysProcess();
    void addNewTask(SysModelTask *newTask, int32_t taskPriority = -1); //!< class method
    void selfInitProcess(); //!< class method
    void resetProcess(uint64_t currentTime); //!< class method
    void reInitProcess(); //!< class method
    void enableProcess() {this->processActive = true;} //!< class method
    void disableProcess() {this->processActive = false;} //!< class method
    void scheduleTask(ModelScheduleEntry & taskCall); //!< class method
    void setProcessName(std::string newName){this->processName = newName;} //!< class method
    std::string getProcessName() { return(processName);} //!< class method
    uint64_t getNextTime() { return(this->nextTaskTime);} //!< class method
    void singleStepNextTask(uint64_t currentNanos); //!< class method
    bool processEnabled() {return this->processActive;} //!< class method
	void changeTaskPeriod(std::string taskName, uint64_t newPeriod); //!< class method
    void setPriority(int64_t newPriority) {this->processPriority = newPriority;} //!< class method
    void disableAllTasks(); //!< class method
    void enableAllTasks(); //!< class method
    bool getProcessControlStatus() {return this->processOnThread;} //!< Allows caller to see if this process is parented by a thread
    void setProcessControlStatus(bool processTaken) {processOnThread = processTaken;} //!< Provides a mechanism to say that this process is allocated to a thread
    
public:
    std::vector<ModelScheduleEntry> processTasks;  //!< -- Array that has pointers to all process tasks
    uint64_t nextTaskTime;  //!< [ns] time for the next Task
    uint64_t prevRouteTime;  //!< [ns] Time that interfaces were previously routed
    std::string processName;  //!< -- Identifier for process
	bool processActive;  //!< -- Flag indicating whether the Process is active
	bool processOnThread; //!< -- Flag indicating that the process has been added to a thread for execution
    int64_t processPriority;  //!< [-] Priority level for process (higher first)
    BSKLogger bskLogger;                      //!< -- BSK Logging
};

#endif /* _SysProcess_H_ */
