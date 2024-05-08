Executive Summary
-----------------
This module profiles a :ref:`PrescribedRotationMsgPayload` message for a specified 2 DOF rotation
for a secondary rigid body connected to a rigid spacecraft hub at a hub-fixed location, :math:`\mathcal{M}`. The body
frame for the prescribed body is designated by the frame :math:`\mathcal{F}`. Accordingly, the prescribed states for the
secondary body are written with respect to the mount frame, :math:`\mathcal{M}`. The prescribed states profiled
in this module are: ``omega_FM_F``, ``omegaPrime_FM_F``, and ``sigma_FM``.

It should be noted that although the inputs to this module are two consecutive rotation angles and axes, the resulting
rotation that is profiled is a 1 DOF rotation. The module converts two given reference angles and their corresponding
rotation axes for the rotation to a single 1 DOF rotation for the rotation. Simple Principal Rotation Vector (PRV)
addition is used on the two given reference PRVs to determine the single PRV required for the rotation.

To use this module for prescribed motion, it must be connected to the :ref:`PrescribedMotionStateEffector`
dynamics module in order to profile the rotational states of the secondary body. A second kinematic profiler module
must also be connected to the prescribed motion dynamics module to profile the translational states of the prescribed
body. The required rotation is determined from the user-specified scalar maximum angular acceleration for the rotation,
:math:`\alpha_{\text{max}}`, the spinning body's initial attitude with respect to the mount frame as the Principal
Rotation Vector ``prv_F0M`` :math:`(\Phi_0, \hat{\boldsymbol{e}}_0)`, and two reference Principal Rotation Vectors
for the rotation, ``prv_F1F0`` :math:`(\Phi_{1a}, \hat{\boldsymbol{e}}_{1a})` and ``prv_F2F1``
:math:`(\Phi_{1b}, \hat{\boldsymbol{e}}_{1b})`.

The maximum scalar angular acceleration is applied constant and positively for the first half of the rotation and
constant negatively for the second half of the rotation. The resulting angular velocity of the prescribed body is
linear, approaching a maximum magnitude halfway through the rotation and ending with zero residual velocity.
The corresponding angle the prescribed body moves through during the rotation is parabolic in time.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  
The module msg connection is set by the user from python.  
The msg type contains a link to the message structure definition, while the description 
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - spinningBodyRef1InMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - input msg with the scalar spinning body rotational reference states for the first rotation
    * - spinningBodyRef2InMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - input msg with the scalar spinning body rotational reference states for the second rotation
    * - prescribedMotionOutMsg
      - :ref:`PrescribedRotationMsgPayload`
      - output message with the prescribed spinning body rotational states


Detailed Module Description
---------------------------
This 2 DOF rotational motion kinematic profiler module converts a given 2 DOF rotation to a single 1 DOF rotation
and profiles the required spinning body rotational motion with respect to a body-fixed mount frame for the
rotation. The inputs to the profiler are the maximum angular acceleration for the rotation,
:math:`\alpha_{\text{max}}`, the spinning body's initial attitude with respect to the mount frame as the Principal
Rotation Vector ``prv_F0M`` :math:`(\Phi_0, \hat{\boldsymbol{e}}_0)`, and two reference Principal Rotation Vectors
for the rotation, ``prv_F1F0`` :math:`(\Phi_{1a}, \hat{\boldsymbol{e}}_{1a})` and ``prv_F2F1``
:math:`(\Phi_{1b}, \hat{\boldsymbol{e}}_{1b})`.

The module first converts the two given reference PRVs to a single PRV, ``prv_F2M`` that represents the final spinning
body attitude with respect to the body-fixed mount frame:

.. math::
    \Phi_2 = 2 \cos^{-1} \left ( \cos \frac{\Phi_{1a}}{2} \cos \frac{\Phi_{1b}}{2} - \sin \frac{\Phi_{1a}}{2} \sin \frac {\Phi_{1b}}{2} \ \hat{\boldsymbol{e}}_{1a} \cdot \hat{\boldsymbol{e}}_{1b} \right )

.. math::
    \hat{\boldsymbol{e}}_2 = \frac{\cos \frac{\Phi_{1b}}{2} \sin \frac{\Phi_{1a}}{2} \ \hat{\boldsymbol{e}}_{1a} + \cos \frac{\Phi_{1a}}{2} \sin \frac{\Phi_{1b}}{2} \ \boldsymbol{e}_{1b} + \sin \frac{\Phi_{1a}}{2} \sin \frac{\Phi_{1b}}{2} \ \hat{\boldsymbol{e}}_{1a} \times \hat{\boldsymbol{e}}_{1b} }{\sin \frac{\Phi_2}{2}}

Subtracting the initial Principal Rotation Vector ``prv_F0M`` from the found reference PRV ``prv_F2M`` gives the
required PRV for the rotation, ``prv_F2F0``:

.. math::
    \Phi_{\text{ref}} = \Delta \Phi = 2 \cos^{-1} \left ( \cos \frac{\Phi_2}{2} \cos \frac{\Phi_0}{2} + \sin \frac{\Phi_2}{2} \sin \frac {\Phi_0}{2} \ \hat{\boldsymbol{e}}_2 \cdot \hat{\boldsymbol{e}}_0 \right )

.. math::
    \hat{\boldsymbol{e}}_3 = \frac{\cos \frac{\Phi_0}{2} \sin \frac{\Phi_2}{2} \ \hat{\boldsymbol{e}}_2 - \cos \frac{\Phi_2}{2} \sin \frac{\Phi_0}{2} \ \hat{\boldsymbol{e}}_0 + \sin \frac{\Phi_2}{2} \sin \frac{\Phi_0}{2} \ \hat{\boldsymbol{e}}_2 \times \hat{\boldsymbol{e}}_0 }{\sin \frac{\Delta \Phi}{2}}

Note that the initial PRV angle, :math:`\Phi_0` is reset to zero for consecutive rotations so that the
reference PRV angle, :math:`\Phi_{\text{ref}}` is always taken as the full angle to be swept during the rotation.

During the first half of the rotation, the spinning body is constantly accelerated with the given maximum
angular acceleration. The spinning body's angular velocity increases linearly during the acceleration phase and reaches
a maximum magnitude halfway through the rotation. The switch time, :math:`t_s` is the simulation time halfway
through the rotation:

.. math::
    t_s = t_0 + \frac{\Delta t}{2}

where the time required for the rotation, :math:`\Delta t` is determined using the found PRV angle for the rotation:

.. math::
    \Delta t = t_f - t_0 = 2\sqrt{ \Phi_{\text{ref}} / \alpha_{\text{max}}}

The resulting trajectory of the angle :math:`\Phi` swept during the first half of the rotation is quadratic. The
profiled motion is concave upwards if the reference angle, :math:`\Phi_{\text{ref}}` is greater than zero. If the
reference angle is negative, the profiled motion is instead concave downwards. The described motion during the first
half of the rotation is characterized by the expressions:

.. math::
    \ddot{\Phi}(t) = \alpha_{\text{max}}

.. math::
    \dot{\Phi}(t) = \alpha_{\text{max}} (t - t_0) + \dot{\Phi}(t_0)

.. math::
    \Phi(t) = a (t - t_0)^2

where

.. math::
    a = \frac{ \frac{1}{2} \Phi_{\text{ref}}}{(t_s - t_0)^2}

Similarly, the second half of the rotation decelerates the spinning body constantly until it reaches a
non-rotating state. The spinning body's angular velocity decreases linearly from its maximum magnitude back to zero.
The trajectory swept during the second half of the rotation is quadratic and concave downwards if the reference angle,
:math:`\Phi_{\text{ref}}` is greater than zero. If the reference angle is negative, the profiled motion is instead
concave upwards. The described motion during the second half of the rotation is characterized by the
expressions:

.. math::
    \ddot{\Phi}(t) = -\alpha_{\text{max}}

.. math::
    \dot{\Phi}(t) = \alpha_{\text{max}} (t - t_f)

.. math::
    \Phi(t) = b (t - t_f)^2  + \Phi_{\text{ref}}

where

.. math::
    b = \frac{ \frac{1}{2} \Phi_{\text{ref}}}{(t_s - t_f)^2}


Module Testing
^^^^^^^^^^^^^^
The unit test for this module simulates TWO consecutive 2 DOF rotational attitude maneuvers for a secondary rigid body
connected to a rigid spacecraft hub. Two maneuvers are simulated to ensure that the module correctly updates the
required relative PRV attitude when a new attitude reference message is written. The unit test checks that the prescribed
body's MRP attitude converges to both reference attitudes for a series of initial and reference attitudes and
maximum angular accelerations. (``sigma_FM_Final1`` is checked to converge to ``sigma_FM_Ref1``, and
``sigma_FM_Final2`` is checked to converge to ``sigma_FM_Ref2``). Additionally, the prescribed body's final angular
velocity magnitude ``thetaDot_Final`` is checked for convergence to the reference angular velocity magnitude,
``thetaDot_Ref``.


User Guide
----------
The user-configurable inputs to the profiler are the maximum angular acceleration for the rotation,
:math:`\alpha_{\text{max}}`, the spinning body's initial attitude with respect to the mount frame as the Principal
Rotation Vector ``prv_F0M`` :math:`(\Phi_0, \hat{\boldsymbol{e}}_0)`, and two reference Principal Rotation Vectors for
the rotation, ``prv_F1F0`` :math:`(\Phi_{1a}, \hat{\boldsymbol{e}}_{1a})` and ``prv_F2F1``
:math:`(\Phi_{1b}, \hat{\boldsymbol{e}}_{1b})`.

This module provides a single output message in the form of :ref:`prescribedRotationMsgPayload`. This prescribed
motion output message can be connected to the :ref:`prescribedMotionStateEffector` dynamics module to directly profile
a state effector's rotational motion. Note that a separate translational profiler module must also be connected to
the prescribed motion dynamics module to fully define the kinematic motion of the prescribed body.

This section is to outline the steps needed to setup a prescribed 2 DOF rotational module in python using Basilisk.

#. Import the prescribedRot1DOF class::

    from Basilisk.fswAlgorithms import prescribedRot2DOF

#. Create an instantiation of a prescribed rotational 2 DOF C module and the associated C++ container::

    PrescribedRot2DOF = prescribedRot2DOF.prescribedRot2DOF()
    PrescribedRot2DOF.ModelTag = "PrescribedRot2DOF"

#. Define all of the configuration data associated with the module. For example::

    rotAxis1_M = np.array([0.0, 1.0, 0.0])                                      # Rotation axis for the first reference rotation angle, thetaRef1a
    rotAxis2_F1 = np.array([0.0, 0.0, 1.0])                                     # Rotation axis for the second reference rotation angle, thetaRef2a
    PrescribedRot2DOF.rotAxis1_M = rotAxis1_M
    PrescribedRot2DOF.rotAxis2_F1 = rotAxis2_F1
    PrescribedRot2DOF.phiDDotMax = phiDDotMax
    PrescribedRot2DOF.omega_FM_F = np.array([0.0, 0.0, 0.0])              # [rad/s] Angular velocity of frame F relative to frame M in F frame components
    PrescribedRot2DOF.omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])         # [rad/s^2] B frame time derivative of omega_FB_F in F frame components
    PrescribedRot2DOF.sigma_FM = np.array([0.0, 0.0, 0.0])                # MRP attitude of frame F relative to frame M

The user is required to set the above configuration data parameters, as they are not initialized in the module.

#. Make sure to connect the required messages for this module.

#. Add the module to the task list::

    unitTestSim.AddModelToTask(unitTaskName, PrescribedRot2DOF)

