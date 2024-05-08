Executive Summary
-----------------

This module  computes the RW motor voltage from the command torque.

The module
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/rwMotorVoltage/_Documentation/Basilisk-rwMotorVoltage-20170113.pdf>`.
contains further information on this module's function,
how to run it, as well as testing.


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
    * - voltageOutMsg
      - :ref:`ArrayMotorVoltageMsgPayload`
      - RW motor voltage output message
    * - torqueInMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - commanded RW motor torque input message
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - RW array configuration input message
    * - rwAvailInMsg
      - :ref:`RWAvailabilityMsgPayload`
      - (optional) RW device availability message
    * - rwSpeedInMsg
      - :ref:`RWSpeedMsgPayload`
      - (optional) RW device speed message

 

