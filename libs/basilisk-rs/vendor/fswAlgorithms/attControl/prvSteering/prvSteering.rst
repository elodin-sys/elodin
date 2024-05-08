Executive Summary
-----------------

This module implements a PRV steering attitude control routine.

The module
:download:`PDF Description </../../src/fswAlgorithms/attControl/prvSteering/_Documentation/AVS-Sim-PRV_Steering-2016-0108.pdf>`
contains further information on this module's function,
how to run it, as well as testing.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.



.. table:: Module I/O Messages
    :widths: 35 35 100

    +-----------------------+-----------------------------------+---------------------------------------------------+
    | Msg Variable Name     | Msg Type                          | Description                                       |
    +=======================+===================================+===================================================+
    | guidInMsg             | :ref:`AttGuidMsgPayload`          | Attitude guidance input message.                  |
    +-----------------------+-----------------------------------+---------------------------------------------------+
    | rateCmdOutMsg         | :ref:`RateCmdMsgPayload`          | Rate command output message.                      |
    +-----------------------+-----------------------------------+---------------------------------------------------+
