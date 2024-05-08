Executive Summary
-----------------

This module computes the desired Body frame torque to dump the momentum in the reaction wheels.

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
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - input message for RW parameters
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - input message for RW speeds
    * - tauMtbRequestOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - output message containing control torque in the Body frame

Detailed Module Description
---------------------------
The following presents the mathematics for computing the torque to be requested by the magnetic torque rods to drive the net momentum of the reaction wheels to zero.

Assume the spacecraft contains :math:`N_{\text{RW}}` RWs. The net RW angular momentum is given by

.. math::
    {}^{\cal B} {\bf h}_{\text{wheels}} = \sum_{i=1}^{N_{\text{RW}}} \hat{\bf g}_{s_i} J_{s_i} \Omega_i

where :math:`\hat{\bf g}_{s_i}` is the RW spin axis in the Body frame :math:`\cal B`, :math:`J_{s_i}`
is the spin axis RW inertia and :math:`\Omega_i` is the RW speed rate about this axis.
The desired torque to be produced by the torque rods to drive the wheel momentum to zero is
then given by the proportional control law

.. math::
    {}^{\cal B} {\pmb\tau}_{\text{desired}} = - K_p \ {}^{\cal B} {\bf h}_{\text{wheels}}

where :math:`K_p` is the proportional feedback gain with units of 1/s.

User Guide
----------
See the example script :ref:`scenarioMtbMomentumManagementSimple` for an illustration on how to use this module. Note that the user must set the momentum dumping gain value ``Kp`` to a postive value and ``GsMatrix_B`` must be specified in column major format.
