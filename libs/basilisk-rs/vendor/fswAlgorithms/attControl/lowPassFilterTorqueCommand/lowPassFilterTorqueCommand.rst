Executive Summary
-----------------

Module applies a low pass filter to the attitude control torque command.

The module
:download:`PDF Description </../../src/fswAlgorithms/attControl/lowPassFilterTorqueCommand/_Documentation/AVS-Sim-LowPassFilterControlTorque-20160108.pdf>`
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
    * - cmdTorqueOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - filtered commanded torque output message
    * - cmdTorqueInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - un-filtered commanded torque input message


