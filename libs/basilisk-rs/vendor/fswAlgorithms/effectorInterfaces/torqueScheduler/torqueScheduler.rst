Executive Summary
-----------------

This module schedules two control torques such that they can be applied simultaneously, one at the time, or neither is applied, and combines them into one output msg. 
This is useful in the case of a system with two coupled degrees of freedom, where the changes in one controlled variable can affect the other controlled variable and thus cause the system to not converge. 


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - motorTorqueOutMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - Output Array Motor Torque Message.
    * - effectorLockOutMsg
      - :ref:`ArrayEffectorLockMsgPayload`
      - Output Array Motor Torque Message.
    * - motorTorque1InMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - #1 Input Array Motor Torque Message.
    * - motorTorque2InMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - #2 Input Array Motor Torque Message. 


Module Assumptions and Limitations
----------------------------------
The two input torques are always read and combined into a single ``motorTorqueOutMsg``. The logic to enable locking the effector or, on the contrary, reading the torque and applying it, is contained into the ``effectorLockOutMsg``.


Detailed Module Description
---------------------------
This module receives a ``lockFlag`` and a a ``tSwitch`` parameter from the user. The first is used to decide how the input torques should be passed to the output torque message. If a torque is to be applied, its corresponding ``effectorLockOutMsg.effectorLockFlag`` is set to zero. If not, its corresponding ``effectorLockOutMsg.effectorLockFlag`` is set to 1. The following cases are possible:

  - ``lockFlag = 0``: both motor torques are applied simultaneously;
  - ``lockFlag = 1``: first motor torque is applied for ``t < tSwitch``, second motor torque is applied for ``t > tSwitch``;
  - ``lockFlag = 2``: second motor torque is applied for ``t < tSwitch``, first motor torque is applied for ``t > tSwitch``;
  - ``lockFlag = 3``: neither of the motor torques are applied. 


User Guide
----------
The required module configuration is::

    scheduler = torqueScheduler.torqueScheduler()
    scheduler.ModelTag = "torqueScheduler"
    scheduler.lockFlag = lockFlag
    scheduler.tSwitch = tSwitch
    unitTestSim.AddModelToTask(unitTaskName, scheduler)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 34 66
   :header-rows: 1

   * - Parameter
     - Description
   * - ``lockFlag``
     - flag to choose the logic according to which the motor torques are applied. If not provided, it defaults to zero.
   * - ``tSwitch``
     - time at which the torque is switched from input 1 to input 2. If not provided it defaults to zero, therefore only input torque is passed.