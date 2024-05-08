Executive Summary
-----------------
This module provides a navigation solution for a spacecraft about a small body. A hybrid extended Kalman filter
estimates relative spacecraft position and velocity with respect to the small body and the attitude and attitude rate of
the asteroid's body frame with respect to the inertial frame.

This module is only meant to provide a somewhat representative autonomous small body proximity operations navigation solution
for POMDP solvers. Therefore, realistic measurement modules do not exist to support this module, and
not every source of uncertainty in the problem is an estimated parameter. Future work will build upon this filter.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  
The module msg connection is set by the user from python.  
The msg type contains a link to the message structure definition, while the description 
provides information on what this message is used for.

.. _ModuleIO_smallBodyNavEKF:
.. figure:: /../../src/fswAlgorithms/smallBodyNavigation/smallBodyNavEKF/_Documentation/Images/moduleIOSmallBodyNavigation.svg
    :align: center

    Figure 1: ``smallBodyNavEKF()`` Module I/O Illustration

Note that this C++ FSW module provides both C- and C++-wrapped output messages.  The regular C++ wrapped output
messages end with the usual ``...OutMsg``.  The C wrapped output messages have the same payload type, but end
with ``...OutMsgC``.  

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - navTransInMsg
      - :ref:`NavTransMsgPayload`
      - Translational nav input message
    * - navAttInMsg
      - :ref:`NavAttMsgPayload`
      - Attitude nav input message
    * - asteroidEphemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - Small body ephemeris input message
    * - sunEphemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - Sun ephemeris input message
    * - thrusterInMsgs
      - :ref:`THROutputMsgPayload`
      - Vector of thruster input messages
    * - cmdForceBodyInMsg
      - :ref:`cmdForceBodyMsgPayload`
      - Commanded force input
    * - navTransOutMsg
      - :ref:`NavTransMsgPayload`
      - Translational nav output message
    * - navTransOutMsgC
      - :ref:`NavTransMsgPayload`
      - C-wrapped translational nav output message
    * - smallBodyNavOutMsg
      - :ref:`SmallBodyNavMsgPayload`
      - Small body nav output msg - states and covariances
    * - smallBodyNavOutMsgC
      - :ref:`SmallBodyNavMsgPayload`
      - C-wrapped small body nav output msg - states and covariances
    * - asteroidEphemerisOutMsg
      - :ref:`EphemerisMsgPayload`
      - Small body ephemeris output message
    * - asteroidEphemerisOutMsgC
      - :ref:`EphemerisMsgPayload`
      - C-wrapped small body ephemeris output message

Detailed Module Description
---------------------------
General Function
^^^^^^^^^^^^^^^^
The ``smallBodyNavEKF()`` module provides a state estimate for a spacecraft in proximity of a small body. The
relative spacecraft position and velocity and small body attitude and rate are estimated
by the filter. The filter assumes full observability of each state. The "measurements" are typically messages written
out by :ref:`simpleNav` and :ref:`planetNav` modules. However, future developers can implement measurement models
that adhere to the required I/O format. The full state vector may be found below:

.. math::
    :label: eq:x_hat

    \mathbf{X} =
    \begin{bmatrix}
    \mathbf{x}_1\\
    \mathbf{x}_2\\
    \mathbf{x}_3\\
    \mathbf{x}_4
    \end{bmatrix}=
    \begin{bmatrix}
    {}^O\mathbf{r}_{S/O} \\
    {}^O\dot{\mathbf{r}}_{S/O} \\
    \boldsymbol{\sigma}_{AN} \\
    {}^A\boldsymbol{\omega}_{AN}
    \end{bmatrix}

The associated frame definitions may be found in the following table.

.. list-table:: Frame Definitions
    :widths: 25 25
    :header-rows: 1

    * - Frame Description
      - Frame Definition
    * - Small Body Hill Frame
      - :math:`O: \{\hat{\mathbf{o}}_1, \hat{\mathbf{o}}_2, \hat{\mathbf{o}}_3\}`
    * - Small Body Body Frame
      - :math:`A: \{\hat{\mathbf{a}}_1, \hat{\mathbf{a}}_2, \hat{\mathbf{a}}_3\}`
    * - Spacecraft Body Frame
      - :math:`B: \{\hat{\mathbf{b}}_1, \hat{\mathbf{b}}_2, \hat{\mathbf{b}}_3\}`
    * - J2000 Inertial Frame
      - :math:`N: \{\hat{\mathbf{n}}_1, \hat{\mathbf{n}}_2, \hat{\mathbf{n}}_3\}`

Initialization
^^^^^^^^^^^^^^

Algorithm
^^^^^^^^^^
This module employs a hybrid extended Kalman filter (EKF) to estimate the relevant states. First, :math:`\hat{\mathbf{x}}_0`
and :math:`P_0` are initialized by the user. The dynamics matrix :math:`A_0` is initialized to identity by the module.

.. math::
    :label: eq:init_x

    \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_0

.. math::
    :label: eq:init_covar

    P_k = P_0

The apriori state estimate :math:`\hat{\mathbf{x}}_{k+1}^-` and STM :math:`\Phi(k+1,k)` are propagated using the equations below:

.. math::
    :label: eq:predict_state

    \dot{\hat{\mathbf{x}}}_{k} = f(\hat{\mathbf{x}}_k, \mathbf{u}_k, w_k, t_k)

.. math::
    :label: eq:stm_dot

    \dot{\Phi}(k+1,k) = A_{k+1}\Phi(k+1,k)

The apriori estimation error covariance :math:`P_{k+1}^-` is then computed by propagating the equations below:

.. math::
    :label: eq:predict_covar

    P_{k+1}^- = \Phi(k+1, k) P_k^+ \Phi(k+1, k)^T + L_kQ_kL_k^T

The measurements are read into the module and the state and covariance are updated using the equation below. If no
new measurements are present, the filter skips this step and writes out the apriori state estimate and covariance.

.. math::
    :label: eq:kalman_gain

    K_{k+1} = P_{k+1}^-H_{k+1}^T(H_{k+1}P_{k+1}^-H_{k+1}^T + M_{k+1}R_{k+1}M_{k+1}^T)^{-1}

.. math::
    :label: eq:update_state

    \hat{\mathbf{x}}_{k+1}^+ = \hat{\mathbf{x}}_{k+1}^- + K_{k+1}[y_{k+1} - h(\hat{\mathbf{x}}_{k+1}^-, v_{k+1}, t_{k+1})]

.. math::
    :label: eq:update_covar

    P_{k+1}^+ = (I-K_{k+1}H_{k+1})P_{k+1}^-(I-K_{k+1}H_{k+1})^T+K_{k+1}M_{k+1}R_{k+1}M_{k+1}^TK_{k+1}^T

The dynamics for each element of :math:`f(\hat{\mathbf{x}}_k, \mathbf{u}_k, w_k, t_k)` may  be found below. The relative
position and velocity dynamics are described in detail by `Takahashi <https://doi.org/10.2514/1.G005733>`__ and
`Scheeres <http://dx.doi.org/10.2514/1.57247>`__. The equations for attitude dynamics are described in detail in Chapters 3 and 4
of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__. We assume that the small body rotates at a
constant rate.

.. math::
    :label: eq:smn_x_dot_1

    \dot{\mathbf{x}}_1 = ^O\dot{\mathbf{r}}_{S/O} = \mathbf{x}_2

.. math::
    :label: eq:smn_x_dot_2

    \begin{split}
    \dot{\mathbf{x}}_2 = ^O\ddot{\mathbf{r}}_{S/O} = -\ddot{F}[\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_1 - 2\dot{F}[\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_2 - \dot{F}^2[\tilde{\hat{\mathbf{o}}}_3][\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_1- \dfrac{\mu_a \mathbf{x}_1}{||\mathbf{x}_1||^3} + \dfrac{\mu_s(3{}^O\hat{\mathbf{d}}{}^O\hat{\mathbf{d}}^T-[I_{3 \times 3}])\mathbf{x}_1}{d^3} \\
    + C_{SRP}\dfrac{P_0(1+\rho)A_{sc}}{M_{sc}}\dfrac{(1\text{AU})^2}{d^2}\hat{\mathbf{o}}_1 + \sum_i^I\dfrac{{}^O\mathbf{F}_i}{M_{sc}} + \sum_j^J\dfrac{{}^O\mathbf{F}_j}{M_{sc}}
    \end{split}

.. math::
    :label: eq:smn_x_dot_3

    \dot{\mathbf{x}}_3 = \dot{\boldsymbol{\sigma}}_{A/N} = \dfrac{1}{4} \Bigr [ \Bigr ( 1-||\mathbf{x}_3||^2 \Bigr ) [I_{3 \times 3}] + 2[\tilde{\mathbf{x}}_3] + 2\mathbf{x}_3\mathbf{x}_3^T \Bigr]\mathbf{x}_4

.. math::
    :label: eq:smn_x_dot_4

    \dot{\mathbf{x}}_4 = {}^A\dot{\boldsymbol{\omega}}_{A/N} = \mathbf{0}

Note that the MRP switching is checked following the procedure outlined in `Karlgaard <https://link.springer.com/content/pdf/10.1007/BF03321529.pdf>`__.

The derivation of the state dynamics matrix :math:`A` is not shown here for brevity.


Module Assumptions and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module assumptions and limitations are listed below:

 - The spacecraft's attitude and rate are perfectly known
 - The small body's position and velocity in the inertial frame are perfectly known
 - Please refer to the cited works for specific assumptions about the filter dynamics
 - The matrix :math:`H_{k+1}` is identity

User Guide
^^^^^^^^^^
The user then must set the following module variables:

- ``A_sc``, the area of the spacecraft in :math:`\text{m}^2`
- ``M_sc``, the mass of the spacecraft in kg
- ``mu_ast``, the gravitational constant of the small body in :math:`\text{m}^3/\text{s}^2`
- ``Q``, the process noise covariances
- ``R``, the measurement noise covariance
- ``x_hat_k`` to initialize :math:`x_0`
- ``P_k`` to initialize :math:`P_0`


The user must connect to each input message described in Table 1.