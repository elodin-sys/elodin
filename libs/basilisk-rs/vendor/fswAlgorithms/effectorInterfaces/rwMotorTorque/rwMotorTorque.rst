Executive Summary
-----------------

This module maps a desired torque to control the spacecraft, and maps it to the available wheels using a minimum norm inverse fit.

The optional wheel availability message is used to include or exclude particular reaction wheels from the torque solution.  The desired control torque can be mapped onto particular orthogonal control axes to implement a partial solution for the overall attitude control torque.  More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/rwMotorTorque/_Documentation/Basilisk-rwMotorTorque-20190320.pdf>`.


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
    * - rwMotorTorqueOutMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - RW motor torque output message
    * - vehControlInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - commanded vehicle control torque input message
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - RW array configuration input message
    * - rwAvailInMsg
      - :ref:`RWAvailabilityMsgPayload`
      - (optional) RW device availability message


