Executive Summary
-----------------
This module estimates the location of the center of mass (CM) of the entire spacecraft system based on a series of torque measurements. The module assumes the ability to measure and provide the control torques delivered to the system at steady state, to ensure equilibrium. Torque measurements are provided sequentially, and measurements updates are performed for each new measurement using a sequential, weighted least-squares (LS) algorithm.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages. The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - thrusterConfigBInMsg
      - :ref:`THRConfigMsgPayload`
      - Input message the thrust application point with respect to the origin of the :math:`\mathcal{B}` frame, thrust unit direction vector, and thrust magnitude, all in :math:`\mathcal{B}`-frame components.
    * - intFeedbackTorqueInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - Input message containing the stabilizing integral feedback control torque.
    * - attGuidInMsg
      - :ref:`AttGuidMsgPayload`
      - Input message containing the attitude and angular rates of the body frame with respect to the guidance reference frame.
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - (Optional) Input message containing the real location of the CM of the system. It is used to verify the correctness of the estimated solution.
    * - vehConfigOutMsg
      - :ref:`VehicleConfigMsgPayload`
      - Output message containing the estimated location of the CM of the system.
    * - cmEstDataOutMsg
      - :ref:`CMEstDataMsgPayload`
      - Output message containing the estimated state, state errors, state covariance, pre- and post-fit residuals.


Detailed Module Description
---------------------------
This module assumes the presence of a gimbaled thruster that is periodically actuated in order to change the direction of the thrust vector with respect to the body frame. This is done primarily beacuse one thrust measurement alone is not enough to resolve the three components of the CM location. Moreover, fixing the thrust vector along a certain direction causes momentum to build up on the system if such direction is not aligned with the CM. The relative attitude :math:`\boldsymbol{\sigma}_\mathcal{B/R}` and relative angular rates :math:`\boldsymbol{\omega}_\mathcal{B/R}` of the spacecraft hub with respect to the reference are part of the inputs because measurements are only processed when the attitude and rate errors drop below a certain user-defined threshold:

.. math::
    \epsilon = \sqrt{\left| \sigma_\mathcal{B/R} \right|^2 + \left| \omega_\mathcal{B/R} \right|^2}

This is done because the torque measurement should only be processed when the spacecraft hub has converged to the reference within a certain accuracy, which means that the integral feedback torque has converged to the external perturbation. Measurements updates do not happen when ``attGuidInMsg`` is not written. Whenever the thrust vector is not aligned with the CM, the thruster torque is acting as an external perturbation on the system, which is what is used in this module to infer the location of the CM. During the time when the thruster remains fixed in the body-frame, such disturbance is constant, which aligns with the assumption of a constant unmodeled torque that can be compensated by the integral feedback term in :ref:`mrpFeedback`. :ref:`thrusterPlatformReference` computes the ideal reference angles for a dual-gimbaled thruster platform in order to articulate the thruster in a way that ensures that momentum is constantly dumped and does not cause wheel saturation.

With :math:`\boldsymbol{t}` being the thrust vector and :math:`\boldsymbol{Z}` being the negative of the integral feedback term, the following quantities are defined at each step:

- :math:`[\boldsymbol{C}_n] = [\boldsymbol{\tilde{t}}]` : linear model
- :math:`\boldsymbol{y}_n = \boldsymbol{Z} + [\boldsymbol{\tilde{t}}] \boldsymbol{r}_{T/B}`
- :math:`[\boldsymbol{K}_n]` : optimal gain
- :math:`\boldsymbol{x}_n = \boldsymbol{r}_{C/B}` : CM location estimate
- :math:`[\boldsymbol{P}_n]` : covariance of the state estimate.

At each step, the state and covariance estimates are updated according to the following:

.. math::
    [\boldsymbol{K}_n] = [\boldsymbol{P}_n] [\boldsymbol{C}_n]^T \left( [\boldsymbol{C}_n] [\boldsymbol{P}_n] [\boldsymbol{C}_n]^T + [\boldsymbol{R}] \right)^{-1} \\
    \boldsymbol{x}_{n+1} = \boldsymbol{x}_n + [\boldsymbol{K}_n] \left( \boldsymbol{y}_n - [\boldsymbol{C}_n] \boldsymbol{x}_n \right) \\
    \left[\boldsymbol{P}_{n+1}\right] = \left( [\boldsymbol{I}] - [\boldsymbol{K}_n] [\boldsymbol{C}_n] \right) [\boldsymbol{P}_n]

where :math:`[\boldsymbol{R}]` is the measurement noise covariance.


Module Assumptions and Limitations
----------------------------------
The correct functioning of this module can only be guaranteed as long as multiple, linearly independent torque measurements are being provided. One static measurement is not enough to resolve the CM location.

When additional external disturbances act on the system, the estimated CM location can be affected. In the presence of a biased, unmodeled external torque such as SRP, the estimated location does not coincide with the CM location, but rather the point through which the thruster produces a torque that cancels the SRP effect. Despite the bias in the measurement, this result is still useful combined with :ref:`thrusterPlatformReference` because it guarantees to reach a steady-state equilibrium. See :ref:`scenarioSepMomentumManagement` for an integrated example scenario.

The frequency at which the thruster is articulated needs to be chosen carefully, because holding the thruster fixed for too long can cause reaction wheel saturation, with consequent loss of attitude and inability to estimate the CM location.

More details can be found in `R. Calaon, C. Allard, and H. Schaub, "Continuous Center-Of-Mass Estimation For A Gimbaled Ion-Thruster Equipped Spacecraft" <http://hanspeterschaub.info/Papers/Calaon2023b.pdf>`__.


User Guide
----------
The required module configuration is::

    cmEstimation = thrustCMEstimation.ThrustCMEstimation()
    cmEstimation.ModelTag = "cmEstimator"
    cmEstimation.attitudeTol = 1e-4
    cmEstimation.r_CB_B = [0.01, -0.025, 0.04]
    cmEstimation.P0 = [0.0025, 0.0025, 0.0025]
    cmEstimation.R0 = [1e-9, 1e-9, 1e-9]
    unitTestSim.AddModelToTask(unitTaskName, cmEstimation)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``attitudeTol``
     - 0
     - convergence error :math:`\epsilon`
   * - ``r_CB_B``
     - [0, 0, 0]
     - initial guess on the CM location
   * - ``P0``
     - [0, 0, 0]
     - diagonal elements of the initial state covariance
   * - ``R0``
     - [0, 0, 0]
     - diagonal elements of the measurement noise covariance

