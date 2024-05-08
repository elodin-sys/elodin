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

#ifndef _SysModelTask_HH_
#define _SysModelTask_HH_

#include <vector>
#include <stdint.h>
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/bskLogging.h"

//! Structure used to pair a model and its requested priority
typedef struct {
    int32_t CurrentModelPriority;  //!< The current model priority. Higher goes first
    SysModel *ModelPtr;  //!< The model associated with this priority
}ModelPriorityPair;

//! Class used to group a set of models into one "Task" of execution
class SysModelTask
{
    
public:
    SysModelTask();
    SysModelTask(uint64_t InputPeriod, uint64_t FirstStartTime=0); //!< class method
    ~SysModelTask();
    void AddNewObject(SysModel *NewModel, int32_t Priority = -1);
    void SelfInitTaskList();
    //void CrossInitTaskList();
    void ExecuteTaskList(uint64_t CurrentSimTime);
	void ResetTaskList(uint64_t CurrentSimTime);
    void ResetTask() {this->NextStartTime = this->FirstTaskTime;} //!< Resets the task
	void enableTask() {this->taskActive = true;} //!< Enables the task.  Great comment huh?
	void disableTask() {this->taskActive = false;} //!< Disables the task.  I know.
    void updatePeriod(uint64_t newPeriod);
    void updateParentProc(std::string parent) {this->parentProc = parent;} //!< Allows the system to move task to a different process
    
public:
    std::vector<ModelPriorityPair> TaskModels;  //!< -- Array that has pointers to all task sysModels
    std::string TaskName;  //!< -- Identifier for Task
    std::string parentProc; //!< -- Process that calls this task
    uint64_t NextStartTime;  //!< [ns] Next time to start task
    uint64_t NextPickupTime;  //!< [ns] Next time read Task outputs
    uint64_t TaskPeriod;  //!< [ns] Cycle rate for Task
    uint64_t FirstTaskTime;  //!< [ns] Time to start Task for first time.  After this time the normal periodic updates resume.
	bool taskActive;  //!< -- Flag indicating whether the Task has been disabled
  BSKLogger bskLogger;                      //!< -- BSK Logging
};

#endif /* _SysModelTask_H_ */
