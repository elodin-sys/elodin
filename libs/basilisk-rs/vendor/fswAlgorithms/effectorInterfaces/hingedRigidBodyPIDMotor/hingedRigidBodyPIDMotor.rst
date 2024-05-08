Executive Summary
-----------------

This module implements a simple Proportional-Integral-Derivative (PID) control law to provide the commanded
torque to a :ref:`spinningBodyOneDOFStateEffector`.


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
      - Output Spinning Body Reference Message.
    * - hingedRigidBodyInMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Input Spinning Body Message Message.
    * - hingedRigidBodyRefInMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Input Spinning Body Reference Message Message. 


Module Assumptions and Limitations
----------------------------------
This module is very simple and does not make any assumptions. The only limitations are those inherent to a PID type of control law, here implemented. The type of response (underdamped, 
overdamped, or critically damped) depends on the choice of gains provided as inputs to the module.


Detailed Module Description
---------------------------
For this module to operate, the user needs to provide control gains ``K``, ``P`` and ``I``. Let's define :math:`\theta_R` and :math:`\dot{\theta}_R` the reference angle and angle rate contained in the
``hingedRigidBodyRefInMsg``, and :math:`\theta` and :math:`\dot{\theta}` the current solar array angle and angle rate contained in the ``hingedRigidBodyInMsg``, which is provided as an output of the :ref:`spinningBodyOneDOFStateEffector`. The control torque is obtained as follows:

.. math::
    T = K (\theta_R - \theta) + P (\dot{\theta}_R - \dot{\theta}) + I \int_0^t (\theta_R - \theta) \text{d}\tau.


User Guide
----------
The required module configuration is::

    motor = hingedRigidBodyPIDMotorConfig.hingedRigidBodyPIDMotorConfig()
    motor.ModelTag = "solarArrayPDController"  
    motor.K = K
    motor.P = P
    motor.P = I
    unitTestSim.AddModelToTask(unitTaskName, motor)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 34 66
   :header-rows: 1

   * - Parameter
     - Description
   * - ``K``
     - proportional gain; defaults to 0 if not provided
   * - ``P``
     - derivative gain; defaults to 0 if not provided
   * - ``I``
     - integral gain; defaults to 0 if not provided
