Executive Summary
-----------------

This module computes the appropriate wheel torques and magnetic torque bar dipoles to bias the wheels to their desired speeds. Note that there is no gurantee that the wheel speeds will converge to their bias when less than 3 orthornormal torque rods are in use. In this case it is recommended that a nullspace controller be used to drive the wheels toward their bias. Documentation for the math used in this module can be found in a `paper <http://dx.doi.org/10.2514/1.G000812>`__ by Eric Hogan.

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
    * - mtbParamsInMsg
      - :ref:`MTBArrayConfigMsgPayload`
      - input message for MTB layout
    * - tamSensorBodyInMsg
      - :ref:`TAMSensorBodyMsgPayload`
      - input message for magnetic field sensor data
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - input message for RW speeds
    * - rwMotorTorqueInMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - input message for RW motor torques
    * - mtbCmdOutMsg
      - :ref:`MTBCmdMsgPayload`
      - output message for MTB dipole commands
    * - rwMotorTorqueOutMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - output message for RW motor torques

User Guide
----------
See the example script :ref:`scenarioMtbMomentumManagement` for an illustration on how to use this module.

The user must set the momentum dumping gain value ``cGain`` to a postive value.  The required input messages are described above.
The optional module list ``wheelSpeedBiases`` can be set to desired RW spin rates in terms of rad/sec.  If this is not set,
the default values are zero rates.

Note that the MTB input configuration message variable ``GtMatrix_B`` must be provided in a row major format.

