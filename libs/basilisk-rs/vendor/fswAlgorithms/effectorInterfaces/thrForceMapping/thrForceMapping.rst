Executive Summary
-----------------

This module is reads in a desired attitude control torque vector and maps it onto a set of thrusters.

The module works for both on-pulsing (nominal thruster state is off such as with RCS thrusters) and off-pulsing (nominal thruster state in such as with DV thrusters). More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/thrForceMapping/_Documentation/Basilisk-ThrusterForces-20160627.pdf>`.

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
    * - thrForceCmdOutMsg
      - :ref:`THRArrayCmdForceMsgPayload`
      - thruster force output message
    * - cmdTorqueInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - commanded attitude control torque vector input message
    * - thrConfigInMsg
      - :ref:`THRArrayConfigMsgPayload`
      - Thruster array configuration input message
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - spacecraft configuration input message


