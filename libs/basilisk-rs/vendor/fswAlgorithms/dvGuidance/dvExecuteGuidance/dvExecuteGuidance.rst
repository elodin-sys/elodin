Executive Summary
-----------------

This method takes its own internal variables and creates an output attitude
command to use for burn execution.  It also flags whether the burn should
be happening or not.

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
    * - burnDataInMsg
      - :ref:`DvBurnCmdMsgPayload`
      - Input message that configures the vehicle burn
