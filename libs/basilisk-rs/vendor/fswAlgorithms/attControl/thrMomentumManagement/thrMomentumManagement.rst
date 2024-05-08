Executive Summary
-----------------

This module reads in the Reaction Wheel (RW) speeds, determines the net RW momentum, and then determines the amount of angular momentum that must be dumped.

A separate thruster firing logic module called thrMomentumDumping will later on compute the thruster on cycling. The module
:download:`PDF Description </../../src/fswAlgorithms/attControl/thrMomentumManagement/_Documentation/Basilisk-thrMomentumManagement-20160817.pdf>`
contains further information on this module's function, how to run it, as well as testing.

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
    * - deltaHOutMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - output message with the requested inertial angular momentum change
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - reaction wheel speed input message
    * - rwConfigDataInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - name of the RWA configuration message

