Executive Summary
-----------------
This module provides a general MRP feedback control law.  This 3-axis control can asymptotically track a general
reference attitude trajectory.  The module is setup to work with or without `N` reaction wheels with
general orientation.  If the reaction wheel states are fed into this module, then the resulting RW
gyroscopic terms are compensated for. If the wheel information is not present, then these terms are ignored.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_mrpFeedback:
.. figure:: /../../src/fswAlgorithms/attControl/mrpFeedback/_Documentation/Images/moduleIOMrpFeedback.svg
    :align: center

    Figure 1: ``mrpFeedback()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - cmdTorqueOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - Control torque output message
    * - intFeedbackTorqueOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - Integral feedback control torque output message
    * - guidInMsg
      - :ref:`AttGuidMsgPayload`
      - Attitude guidance input message
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - Vehicle configuration input message
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - Reaction wheel array configuration input message
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - Reaction wheel speeds message
    * - rwAvailInMsg
      - :ref:`RWAvailabilityMsgPayload`
      - Reaction wheel availability message.


Detailed Module Description
---------------------------
General Function
^^^^^^^^^^^^^^^^
The ``mrpFeedback`` module creates the MRP attitude feedback control torque :math:`{\bf L}_{r}` developed in chapter 8 of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.  The input and output messages are illustrated in :ref:`ModuleIO_mrpFeedback`.  The output message is a body-frame control torque vector.  The required attitude guidance message contains both attitude tracking error states as well as reference frame states.  This message is read in with every update cycle. The vehicle configuration message is only read in on reset and contains the spacecraft inertia tensor about the vehicle's center of mass location.

The MRP feedback control can compensate for Reaction Wheel (RW) gyroscopic effects as well.  This is an optional input message where the RW configuration array message contains the RW spin axis :math:`\hat{\bf g}_{s,i}` information and the RW polar inertia about the spin axis :math:`I_{W_{s,i}}`.  This is only read in on reset.  The RW speed message contains the RW speed :math:`\Omega_{i}` and is read in every time step.  The optional RW availability message can be used to include or not include RWs in the MRP feedback.  This allows the module to selectively turn off some RWs.  The default is that all RWs are operational and are included.



Initialization
^^^^^^^^^^^^^^
Simply call the module reset function prior to using this control module.  This will reset the prior function call time variable, and reset the attitude error integral measure.  The control update period :math:`\Delta t` is evaluated automatically.

Algorithm
^^^^^^^^^
This module employs the MRP feedback algorithm of Example (8.14) of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.  This  nonlinear attitude tracking control includes an integral measure of the attitude error.  Further, we seek to avoid quadratic :math:`\pmb\omega` terms to reduce the likelihood of control saturation during a detumbling phase.  Let the new nonlinear feedback control be expressed as

.. math:: [G_{s}]{\bf u}_{s} = -{\bf L}_{r}
    :label: eq:GusRW

where

.. math::
    :label: eq:Lr

    {\bf L}_{r} =  -K \pmb\sigma - [P] \delta\pmb\omega - [P][K_{I}] {\bf z}  - [I_{\text{RW}}](-\dot{\pmb\omega}_{r} + [\tilde{\pmb\omega}]\pmb\omega_{r}) - {\bf L}
    \\
    + ([\tilde{\pmb \omega}_{r}] + [\widetilde{K_{I}{\bf z}}])
    \left([I_{\text{RW}}]\pmb\omega + [G_{s}]{\bf h}_{s} \right)

and

.. math::    h_{s_{i}} = I_{W_{s_{i}}} (\hat{\bf g}_{s_{i}}^{T} \pmb\omega_{B/N} + \Omega_{i})
    :label: eq:hsi

with :math:`I_{W_{s}}` being the RW spin axis inertia.

The integral attitude error measure :math:`\bf z` is defined through

.. math::  {\bf z} = K \int_{t_{0}}^{t} \pmb\sigma \text{d}t + [I_{\text{RW}}](\delta\pmb\omega - \delta\pmb\omega_{0})
    :label: eq:zKi

In the BSK module the vector :math:`\delta\pmb\omega_{0}` is hard-coded to a zero vector.  This function will work for any initial tracking error, and this assumption doesn't impact performance. A limit to the magnitude of the :math:`\int_{t_{0}}^{t} \pmb\sigma \text{d}t` can be specified, which is a scalar compared to each element of the integral term.

The integral measure :math:`\bf z` must be computed to determine :math:`[P][K_{I}] {\bf z}`, and the expression :math:`[\widetilde{K_{I}{\bf z}}]` is added to :math:`[\widetilde{\pmb\omega_{r}}]` term.

To analyze the stability of this control, the following Lyapunov candidate function is used:

.. math::
    :label: eq:V

    V(\delta\pmb\omega, \pmb\sigma, {\bf z}) = \frac{1}{2} \delta\pmb\omega^{T} [I_{\text{RW}}] \delta\pmb\omega
    + 2 K \ln ( 1 + \pmb\sigma^{T} \pmb\sigma) + \frac{1}{2} {\bf z} ^{T} [K_{I}]{\bf z}

provides a convenient positive definite attitude error function.  The attitude feedback gain $K$ is positive, while the integral feedback gain :math:`[K_{I}]` is a symmetric positive definite matrix.
The resulting Lyapunov rate expression is given by

.. math::
    :label: eq:V_dot

    \dot V =  (\delta\pmb\omega + [K_{I}]{\bf z})^{T} \left ( [I_{\text{RW}}] \frac{{}^{\mathcal{B \!}}\text{d}}{\text{d}t} (\delta\pmb\omega) + K \pmb \sigma \right )

Substituting the equations of motion of a spacecraft with :math:`N` reaction wheels (see Eq.~(8.160) in `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__, results in

.. math::
    :label: eq:V_dot2

    \dot V =  (\delta\pmb\omega + [K_{I}]{\bf z} )^{T} \left (
     - [\tilde{\pmb\omega}] ([I_{\text{RW}}] \pmb\omega +[G_{s}]{\bf h}_{s})
    \\
    - [G_{s}] {\bf u}_{s} + {\bf L}
     - [I_{\text{RW}}] ( \dot{\pmb \omega}_{r} - [\tilde{\pmb\omega}]\pmb\omega_{r}) + K \pmb\sigma
    \right)

Substituting the control expression in Eq. :eq:`eq:GusRW` and making use of :math:`\pmb \alpha = \pmb\omega_{r} - [K_{I}]{\bf z}` leads  to

.. math::
    :label: eq:V_dot3

    \dot V &=  (\delta\pmb\omega + [K_{I}]{\bf z} )^{T} \Big (
    - ([\tilde{\pmb\omega}] - [\tilde{\pmb\omega}_{r}] + [\widetilde{K_{I}{\bf z}}]) ([I_{\text{RW}}] \pmb\omega
    + [G_{s}]{\bf h}_{s})
    +( K \pmb\sigma - K \pmb\sigma)
    \\
    & \quad - [P]\delta\pmb\omega - [P][K_{I}]\pmb z + [I_{\text{RW}}](\dot{\pmb\omega}_{r}
    - [\tilde{\pmb\omega}]\pmb\omega_{r}) - [I_{\text{RW}}](\dot{\pmb\omega}_{r} - [\tilde{\pmb\omega}]\pmb\omega_{r})
    + ( {\bf L} - {\bf L})
    \Big)
    \\
    &=  (\delta\pmb\omega + [K_{I}]{\bf z} )^{T} \Big (
     - ([\widetilde{\delta\pmb\omega}] + [\widetilde{K_{I}{\bf z}}] )  ([I_{\text{RW}}] \pmb\omega + [G_{s}]{\bf h}_{s})
     - [P] (\delta\pmb\omega + [K_{I}]{\bf z})
    \Big )

Because :math:`(\delta\pmb\omega + [K_{I}]{\bf z} )^{T}  ([\widetilde{\delta\pmb\omega}] + [\widetilde{K_{I}{\bf z}}] ) = 0`, the Lyapunov rate reduces the negative semi-definite expression

.. math::    \dot V = -  (\delta\pmb\omega + [K_{I}]{\bf z} )^{T} [P]  (\delta\pmb\omega + [K_{I}]{\bf z} )
    :label: eq:V_dot4

This proves the new control is globally stabilizing.  Asymptotic stability is shown following the same steps as for the  nonlinear integral feedback control in Eq. (8.104) in `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.

One of the goals set forth at the beginning of the example was avoiding quadratic :math:`\pmb\omega` feedback terms to reduce the odds of control saturation during periods with large :math:`\pmb\omega` values.  However, the control in Eq. :eq:`eq:GusRW` contains a product of :math:`\bf z` and :math:`\pmb\omega`.  Let us study this term in more detail.  The :math:`\pmb\omega` expression with this product terms is found to be

.. math::
    :label: eq:mrp:1

    [\widetilde{K_{I}{\bf z}}] ([I_{\text{RW}}]\pmb \omega)
     \quad \Rightarrow \quad
    -  (
    [\widetilde{I_{\text{RW}} \pmb \omega}]
     ) ([K_{I}] [I_{\text{RW}}] \pmb \omega + \cdots )

If the integral feedback gain is a scalar :math:`K_{I}`, rather than a symmetric positive definite
matrix :math:`[K_{I}]`, the quadratic :math:`\pmb\omega` term vanishes.  If the
full :math:`3\times 3` gain matrix is employed, then quadratic rate feedback terms are retained.


Module Assumptions and Limitations
----------------------------------
This module assumes the main spacecraft is a rigid body.  If RW devices are installed, their wheel speeds are assumed to be fed into this control solution.


User Guide
----------
This module requires the following variables from the required input messages:

- :math:`{\pmb\sigma}_{B/N}` as ``guidCmdData.sigma_BR``
- :math:`^B{\pmb\omega}_{B/R}`  as ``guidCmdData.omega_BR_B``
- :math:`^B{\pmb\omega}_{R/N}` as ``guidCmdData.omega_RN_B``
- :math:`^B\dot{\pmb\omega}_{R/N}` as ``guidCmdData.domega_RN_B``
- :math:`[I]`, the inertia matrix of the body as ``vehicleConfigOut.ISCPntB_B``

The gains :math:`K` and :math:`P` must be set to positive values.  The integral gain :math:`K_i` is optional, it is a negative number by default. Setting this variable to a negative number disables the error integration for the controller, leaving just PD terms. The integrator is required to maintain asymptotic tracking in the presence of an external disturbing torque.  The ``integralLimit`` is a scalar value applied in an element-wise check to ensure that the value of each element of the :math:`\int_{t_{0}}^{t} \pmb\sigma \text{d}t` vector is within the desired limit. If not, the sign of that element is persevered, but the magnitude is replaced by ``integralLimit``.

If the ``rwParamsInMsg`` is specified, then the associated ``rwSpeedsInMsg`` is required as well.

The ``rwAvailInMsg`` is optional and is used to selectively include RW devices in the control solution.

The ``controlLawType`` is an input that enables the user to choose between two different control laws. When ``controlLawType = 0``, the control law is that specified in :eq:`eq:Lr`. Otherwise, the control law takes the form:

.. math::

    {\bf L}_{r} =  -K \pmb\sigma - [P] \delta\pmb\omega - [P][K_{I}] {\bf z}  - [I_{\text{RW}}](-\dot{\pmb\omega}_{r} + [\tilde{\pmb\omega}]\pmb\omega_{r}) - {\bf L}
    \\
    + [\tilde{\pmb \omega}]
    \left([I_{\text{RW}}]\pmb\omega + [G_{s}]{\bf h}_{s} \right).

This control law is also asymptotically stable. The advantage when compared to :eq:`eq:Lr` is that in this one, the integral control feedback, which may contain integration errors, only appears once. On the downside, this control law depends quadratically on the angular rates of the spacecraft, and could cause a large control torque when the spacecraft is tumbling at a high rate. When unspecified, this parameter defaults to ``controlLawType = 0``.
