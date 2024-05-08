Executive Summary
-----------------

This algorithm is used to control both the RCS and DV thrusters when
executing a trajectory adjustment.


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
    * - rwSpeedInMsg
      - :ref:`RWSpeedMsgPayload`
      - RW speed input message
    * - rwConfigInMsg
      - :ref:`RWConstellationMsgPayload`
      - RW configuration input message
    * - thrConfigInMsg
      - :ref:`THRArrayConfigMsgPayload`
      - Thruster configuration input message
    * - vecConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - spacecraft configuration input message
    * - thrCmdOutMsg
      - :ref:`THRArrayOnTimeCmdMsgPayload`
      - thruster array commanded on time output message


