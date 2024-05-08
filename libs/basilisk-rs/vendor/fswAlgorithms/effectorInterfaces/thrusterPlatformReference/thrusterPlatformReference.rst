Executive Summary
-----------------
This module computes a reference orientation for a dual-gimballed platform connected to the main hub. The platform can only perform a tip-and-tilt type of rotation, and therefore one degree of freedom is blocked. A thruster is mounted on the platform, whose direction is known in platform-frame coordinates. The goal of this module is to compute a reference orientation for the platform which can align the thruster direction with the system's center of mass, to zero the net torque produced by the thruster on the spacecraft. Alternatively, this module can offset the thrust direction with respect to the center of mass to produce a net torque that dumps the momentum accumulated on the reaction wheels.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - Input vehicle configuration message containing the position of the center of mass of the system.
    * - thrusterConfigFInMsg
      - :ref:`THRConfigMsgPayload`
      - Input thruster configuration message containing the thrust direction vector and magnitude in **platform frame coordinates**. The entry ``rThrust_B`` here is the position of the thrust application point, with respect to the origin of the platform frame, in platform-frame coordinates (:math:`{}^\mathcal{F}\boldsymbol{r}_{T/F}`).
    * - rwConfigDataInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - Input message containing the number of reaction wheels, relative inertias and orientations with respect to the body frame.
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - Input message containing the relative speeds of the reaction wheels with respect to the hub.
    * - hingedRigidBodyRef1OutMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Output message containing the reference angle and angle rate for the tip angle.
    * - hingedRigidBodyRef2OutMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Output message containing the reference angle and angle rate for the tilt angle.
    * - bodyHeadingOutMsg
      - :ref:`BodyHeadingMsgPayload`
      - Output message containing the unit direction vector of the thruster in body-frame coordinates.
    * - thrusterTorqueOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - Output message containing the opposite of the net torque produced by the thruster on the system.
    * - thrusterConfigBOutMsg
      - :ref:`THRConfigMsgPayload`
      - Output thruster configuration message containing the thrust direction vector and magnitude in **reference body frame coordinates**. The entry ``rThrust_B`` here is the position of the thrust application point, with respect to the origin of the body frame, in body-frame coordinates (:math:`{}^\mathcal{B}\boldsymbol{r}_{T/B}`).


Detailed Module Description
---------------------------
A detailed mathematical derivation of the equations implemented in this module can be found in `R. Calaon, L. Kiner, C. Allard and H. Schaub, "Momentum Management of a Spacecraft equipped with a Dual-Gimballed Electric Thruster"  <http://hanspeterschaub.info/Papers/Calaon2023a.pdf>`__.

This module computes a direction cosine matrix :math:`[\mathcal{FM}]` that describes the rotation between the platform frame :math:`\mathcal{F}` and the mount frame :math:`\mathcal{M}`. To be compliant with the constraint in the motion of the platform, i.e. the dual gimbal, such frame must have a zero in the element (2,1). When such condition is met, the reference angles computed from the D.C.M. allow to align the thruster through the system's center of mass. The input parameters for this module allow to specify offsets between the origin :math:`M` of the hub-fixed mount frame :math:`\mathcal{M}` and the origin :math:`F` of the platform-fixed frame :math:`\mathcal{F}`, the application point of the thruster force in the :math:`\mathcal{F}` frame, and the direction, in :math:`\mathcal{F}`-frame coordinates, of the thrust vector.

When the optional input messages ``rwConfigDataInMsg`` and ``rwSpeedsInMsg`` the user can specify an input parameter ``K``, which is the proportional gain of a control gain that computes an offset with respect to the center of mass: this allows for the thruster to apply a torque on the system that dumps the momentum accumulated on the wheels. Such control law has the expression:

.. math:: 
    \boldsymbol{d} = -\frac{1}{t^2} \boldsymbol{t} \times(\kappa \boldsymbol{h}_w + \kappa_I \boldsymbol{H}_w)

where :math:`\boldsymbol{h}_w` is the momentum on the wheels and :math:`\boldsymbol{H}_w` the integral over time of the momentum:

.. math::
    \boldsymbol{H}_w = \int_{t_0}^t \boldsymbol{h}_w \text{d}t.

The inputs ``theta1Max`` and ``theta2Max`` are used to set bounds on the output reference angles for the platform. If there are no mechanical bounds, setting these inputs to a negative value bypasses the routine that bounds these angles.

Module Assumptions and Limitations
----------------------------------
As pointed out in the paper referenced above, it is not always guaranteed that a direction cosine matrix exists, that can satisfy both the pointing requirement on the thrust direction and the kinematic constraint on the dual-gimballed platform. When a solution does not exist, a minimum problem is solved to compute the closest constraint-incompliant D.C.M. The tip and tilt referemce angles :math:`\nu_{1R}` and :math:`\nu_{2R}` are extracted from the final D.C.M. according to:

.. math::
    \begin{align}
        \nu_{1R} &= \arctan \left( \frac{f_{23}}{f_{22}} \right) & 
        \nu_{2R} &= \arctan \left( \frac{f_{31}}{f_{11}} \right)
    \end{align}

without checking whether the D.C.M. :math:`[\mathcal{FM}]` is constraint compliant. As a result, the angles :math:`\nu_{1R}` and :math:`\nu_{2R}` produce a constraint compliant reference, which however might not align the thruster with the desired point in the hub.


User Guide
----------
The required module configuration is::

    platformReference = thrusterPlatformReference.thrusterPlatformReference()
    platformReference.ModelTag  = "platformReference"
    platformReference.sigma_MB  = sigma_MB
    platformReference.r_BM_M    = r_BM_M
    platformReference.r_FM_F    = r_FM_F
    platformReference.K         = K
    platformReference.Ki        = Ki
    platformReference.theta1Max = theta1Max
    platformReference.theta2Max = theta2Max
    scSim.AddModelToTaskAddModelToTask(simTaskName, platformReference)
 	
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
    * - ``K``
      - 0
      - proportional gain of the momentum dumping control loop
    * - ``Ki`` (optional)
      - 0
      - integral gain of the momentum dumping control loop
    * - ``theta1Max`` (optional)
      - 0
      - absolute bound on tip angle
    * - ``theta2Max`` (optional)
      - 0
      - absolute bound on tilt angle
