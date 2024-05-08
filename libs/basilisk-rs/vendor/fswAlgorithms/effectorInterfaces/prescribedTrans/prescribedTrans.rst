Executive Summary
-----------------
This module profiles a :ref:`PrescribedTranslationMsgPayload` message for a specified 1 DOF translation
for a secondary prescribed rigid body connected to a rigid spacecraft hub at a hub-fixed location, :math:`\mathcal{M}`.
The body frame for the prescribed body is designated by the frame :math:`\mathcal{F}`. Accordingly, the prescribed
states for the secondary body are written with respect to the mount frame, :math:`\mathcal{M}`. The prescribed states
profiled in this module are: ``r_FM_M``, ``rPrime_FM_M``, and ``rPrimePrime_FM_M``.

To use this module for prescribed motion, it must be connected to the :ref:`PrescribedMotionStateEffector`
dynamics module in order to profile the translational states of the secondary body. A second kinematic profiler
module must also be connected to the prescribed motion dynamics module to profile the rotational states of the
prescribed body. The required translation is determined from the user-specified scalar maximum acceleration
:math:`a_{\text{max}}`, the mount frame axis for the translational motion, the prescribed body's initial position
vector with respect to the mount frame :math:`\boldsymbol{r}_{F/M}(t_0)`, and the reference position vector or the
prescribed body with respect to the mount frame :math:`\boldsymbol{r}_{F/M} (\text{ref})`.

The maximum scalar acceleration is applied constant and positively for the first half of the translation and
constant negatively for the second half of the translation. The resulting velocity of the prescribed body is
linear, approaching a maximum magnitude halfway through the translation and ending with zero residual velocity.
The corresponding translational trajectory the prescribed body moves through during the translation is parabolic in time.

.. warning::
    This module is now deprecated. See the :ref:`PrescribedLinearTranslation` module that replaces this module.

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
    * - linearTranslationRigidBodyInMsg
      - :ref:`LinearTranslationRigidBodyMsgPayload`
      - input msg with the prescribed body reference states
    * - prescribedTranslationOutMsg
      - :ref:`PrescribedTranslationMsgPayload`
      - output message with the prescribed body translational states


Detailed Module Description
---------------------------
This translational motion kinematic profiler module is written to profile a rigid body's translational motion with
respect to a hub-fixed mount frame. The inputs to the profiler are the scalar maximum acceleration for the translation
:math:`a_{\text{max}}`, the mount frame axis for the translational motion, the prescribed body's initial position
vector with respect to the mount frame :math:`\boldsymbol{r}_{F/M}(t_0)`, and the reference position vector or the
prescribed body with respect to the mount frame :math:`\boldsymbol{r}_{F/M} (\text{ref})`.
The magnitudes of the initial and final position vectors are denoted :math:`r_0` and :math:`r_{\text{ref}}`,
respectively. The prescribed body is assumed to be at rest at the beginning of the translation.

Subtracting the initial position from the reference position vector gives the required relative position vector in the
direction of translation:

.. math::
    \Delta \boldsymbol{r} = \boldsymbol{r}_{F/M}(\text{ref}) - \boldsymbol{r}_{F/M}(t_0)

The magnitude of the determined relative position vector gives the required translational distance :math:`\Delta r`.
During the first half of the translation, the prescribed body is constantly accelerated with the given maximum
acceleration. The prescribed body's velocity increases linearly during the acceleration phase and reaches a maximum
magnitude halfway through the translation.

The switch time, :math:`t_s` is the simulation time halfway through the translation:

.. math::
    t_s = t_0 + \frac{\Delta t}{2}

The time required for the translation :math:`\Delta t` is determined using the inputs to the profiler:

.. math::
    \Delta t = \sqrt{\frac{4 r_{\text{ref}} - 8 r_0}{\ddot{a}_{\text{max}}}}

The resulting trajectory of the position vector :math:`r = || \boldsymbol{r}_{F/M} ||_2` magnitude during the first
half of the translation is parabolic. The profiled motion is concave upwards if the reference position magnitude
:math:`r_{\text{ref}}` is greater than the initial position magnitude :math:`r_0`. If the converse is true,
the profiled motion is instead concave downwards. The described motion during the first half of the translation
is characterized by the expressions:

.. math::
    r^{''}_{F / M}(t) = a_{\text{max}}

.. math::
    r^{'}_{F / M}(t) = a_{\text{max}} (t - t_0)

.. math::
    r_{F / M}(t) = c_1 (t - t_0)^2  + r_0

where

.. math::
    c_1 = \frac{r_{\text{ref}} - r_0}{2(t_s - t_0)^2}


Similarly, the second half of the translation decelerates the prescribed body constantly until it reaches the desired
position with zero velocity. The prescribed body velocity decreases linearly from its maximum magnitude back to zero.
The trajectory during the second half of the translation is quadratic and concave downwards if the reference position
magnitude is greater than the initial position magnitude. If the converse is true, the profiled motion is instead
concave upwards. The described motion during the second half of the translation is characterized by the expressions:

.. math::
    r^{''}_{F / M}(t) = -a_{\text{max}}

.. math::
    r^{'}_{F / M}(t) = a_{\text{max}} (t - t_f)

.. math::
    r_{F / M}(t) = c_2 (t - t_f)^2  + r_{\text{ref}}

where

.. math::
    c_2 = \frac{r_{\text{ref}} - r_0}{2 (t_s - t_f)^2}

Module Testing
^^^^^^^^^^^^^^
This unit test for this module ensures that the profiled translation is properly computed for a series of
initial and reference positions and maximum accelerations. The final prescribed position magnitude ``r_FM_M_Final`` and
velocity magnitude ``rPrime_FM_M_Final`` are compared with the reference values ``r_FM_M_Ref`` and
``rPrime_FM_M_Ref``, respectively.

User Guide
----------
The user-configurable inputs to the profiler are the scalar maximum acceleration for the translation
:math:`a_{\text{max}}`, the mount frame axis for the translational motion, the prescribed body's initial position
vector with respect to the mount frame :math:`\boldsymbol{r}_{F/M}(t_0)`, and the reference position vector of the
prescribed body with respect to the mount frame :math:`\boldsymbol{r}_{F/M} (\text{ref})`.

This module provides a :ref:`PrescribedTranslationMsgPayload` output message that can be connected to the
:ref:`PrescribedMotionStateEffector` dynamics module to directly profile a state effector's translational motion.
Note that a separate rotational profiler module can be connected to the prescribed motion dynamics module
to fully define the kinematic motion of the prescribed body.

This section is to outline the steps needed to setup a prescribed translational module in python using Basilisk.

#. Import the prescribedTrans class::

    from Basilisk.fswAlgorithms import prescribedTrans

#. Create an instantiation of a prescribed translational C module and the associated C++ container::

    PrescribedTrans = prescribedTrans.prescribedTrans()
    PrescribedTrans.ModelTag = "prescribedTrans"

#. Define all of the configuration data associated with the module. For example::

    PrescribedTrans.transAxis_M = np.array([1.0, 0.0, 0.0])
    PrescribedTrans.scalarAccelMax = 0.01  # [m/s^2]
    PrescribedTrans.r_FM_M = np.array([0.0, 0.0, 0.0])
    PrescribedTrans.rPrime_FM_M = np.array([0.0, 0.0, 0.0])
    PrescribedTrans.rPrimePrime_FM_M = np.array([0.0, 0.0, 0.0])

The user is required to set the above configuration data parameters, as they are not initialized in the module.

#. Make sure to connect the required messages for this module.

#. Add the module to the task list::

    unitTestSim.AddModelToTask(unitTaskName, PrescribedTrans)

