Executive Summary
-----------------
This module simulates receives the measured tip and tilt angles of the thruster platform, together with the thruster configuration information expressed in platform-frame coordinates. The purpose of this module is to output the thruster configuration information in body-frame coordinates, at any instant in time, given the measured platform states (tip-and-tilt angles).

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the user from python.  The msg type contains a link to the message structure definition, while the description provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - thrusterConfigFInMsg
      - :ref:`THRConfigMsgPayload`
      - Input thruster configuration message containing the thrust direction vector and magnitude in **platform frame coordinates**. The entry ``rThrust_B`` here is the position of the thrust application point, with respect to the origin of the platform frame, in platform-frame coordinates (:math:`{}^\mathcal{F}\boldsymbol{r}_{T/F}`).
    * - hingedRigidBody1InMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Input message containing the current tip angle and tip rate of the platform frame :math:`\mathcal{F}` with respect to the mount frame :math:`\mathcal{M}`.
    * - hingedRigidBody2InMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Input message containing the current tilt angle and tilt rate of the platform frame :math:`\mathcal{F}` with respect to the mount frame :math:`\mathcal{M}`.
    * - thrusterConfigBOutMsg
      - :ref:`THRConfigMsgPayload`
      - Output thruster configuration message containing the thrust direction vector and magnitude in **body frame coordinates**, as a function of tip and tilt angles. The entry ``rThrust_B`` here is the position of the thrust application point, with respect to the origin of the body frame, in body-frame coordinates (:math:`{}^\mathcal{B}\boldsymbol{r}_{T/B}`).



Detailed Module Description
---------------------------
This module reads the tip and tilt angles from the :ref:`spinningBodyTwoDOFStateEffector` that simulates the platform. These angles allow to define the direction cosine matrix :math:`[\mathcal{FM}]` that describes the rotation between the platform frame :math:`\mathcal{F}` with respect to the mount frame :math:`\mathcal{M}`. Accounting for the known offsets between the two frames, as well as the offset between the thruster application point and the origin of the frame :math:`\mathcal{F}`, this module outputs the thrust direction vector in body-frame coordinates :math:`{}^\mathcal{B}\hat{t}` and the position of the thrust application point with respect to the origin of the body frame :math:`{}^\mathcal{B}r_{T/B}`.

A more detailed description of the thruster-platform assembly can be found in `R. Calaon, L. Kiner, C. Allard and H. Schaub, "Momentum Management of a Spacecraft equipped with a Dual-Gimballed Electric Thruster"  <http://hanspeterschaub.info/Papers/Calaon2023a.pdf>`__ and in :ref:`thrusterPlatformReference`.


User Guide
----------
The required module configuration is::

    platformState = thrusterPlatformState.thrusterPlatformState()
    platformState.ModelTag = "platformState"
    platformState.sigma_MB = sigma_MB
    platformState.r_BM_M = r_BM_M
    platformState.r_FM_F = r_FM_F
    scSim.AddModelToTaskAddModelToTask(simTaskName, platformState)
 	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
    :widths: 25 25 50
    :header-rows: 1

    * - Parameter
      - Default
      - Description
    * - ``sigma_MB``
      - [0, 0, 0]
      - relative rotation between body-fixed frames :math:`\mathcal{M}` and :math:`\mathcal{B}`
    * - ``r_BM_M``
      - [0, 0, 0]
      - relative position of point :math:`B` with respect to point :math:`M`, in :math:`\mathcal{M}`-frame coordinates
    * - ``r_FM_F``
      - [0, 0, 0]
      - relative position of point :math:`F` with respect to point :math:`M`, in :math:`\mathcal{F}`-frame coordinates
