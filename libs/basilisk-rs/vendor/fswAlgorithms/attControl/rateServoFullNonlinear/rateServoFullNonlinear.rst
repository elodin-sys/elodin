Executive Summary
-----------------

This module implements a nonlinear rate servo control uses the attiude steering message and determine the ADCS control torque vector.

The module
:download:`PDF Description </../../src/fswAlgorithms/attControl/rateServoFullNonlinear/_Documentation/AVS-Sim-nonlinRateServo-2019-0327.pdf>`
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
      - commanded torque output message
    * - guidInMsg
      - :ref:`AttGuidMsgPayload`
      - attitude guidance input message
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - vehicle configuration input message
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - (optional) RW speed input message
    * - rwAvailInMsg
      - :ref:`RWAvailabilityMsgPayload`
      - (optional) RW availability input message
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - (optional) RW configuration parameter input message
    * - rateSteeringInMsg
      - :ref:`RateCmdMsgPayload`
      - commanded rate input message

