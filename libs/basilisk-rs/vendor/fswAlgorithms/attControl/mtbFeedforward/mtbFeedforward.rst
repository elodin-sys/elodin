Executive Summary
-----------------

This module computes the torque produced by the magnetic torque bars and applies it to the current commanded body torque as a feedforward term.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages. The module msg connection is set by the
user from python. The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - vehControlInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - input message containing the current control torque in the Body frame
    * - dipoleRequestMtbInMsg
      - :ref:`MTBCmdMsgPayload`
      - input message containing the individual dipole requests for each torque bar on the vehicle
    * - tamSensorBodyInMsg
      - :ref:`TAMSensorBodyMsgPayload`
      - input message for magnetic field sensor data
    * - mtbParamsInMsg
      - :ref:`MTBArrayConfigMsgPayload`
      - input message for MTB layout
    * - vehControlOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - output message containing the current control torque in the Body frame

Detailed Module Description
---------------------------

The following presents the mathematics for computing the expected torque produced by the torque rods given their alignment matrix, Gt, the individual rod dipole commands, and the local magnetic field vector in the Body frame. This value is then fed forward to reaction wheels by adding it to the current command Body frame torque to drive the net momentum of the reaction wheel system to zero.

The expected torque produced by the torque rods is given by

.. math::
    {}^{\cal B} {\pmb \tau}_{\text{rods}} = [G_t] {\pmb\mu}_{\text{cmds}} \times \ {}^{\cal B}{\bf b}

and the feed forward command used to dump the momentum of the reaction wheels is simply the
negation of the expected torque produced by the rods.

.. math::
    {}^{\cal B} {\pmb\tau}_{\text{ff}} = - {}^{\cal B}{\pmb\tau}_{\text{rods}}

User Guide
----------
See the example script :ref:`scenarioMtbMomentumManagementSimple` for an illustration on how to use this module. Note that the user must specify the torque rod alignment matrix ``GtMatrix_B`` in row major form.
