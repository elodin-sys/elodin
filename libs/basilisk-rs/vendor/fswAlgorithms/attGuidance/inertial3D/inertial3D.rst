Executive Summary
-----------------

This attitude guidance module create a reference attitude message that points in fixed inertial direction. The module
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/inertial3D/_Documentation/Basilisk-Inertial3D-2016-01-15.pdf>`
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
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference output message

