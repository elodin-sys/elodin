Executive Summary
-----------------

This method takes the estimated body-observed sun vector and computes the current attitude/attitude rate errors to pass on to control.

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
    * - cmdTorqueBodyInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - attitude reference output message
    * - thrOnTimeOutMsg
      - :ref:`THRArrayOnTimeCmdMsgPayload`
      - thruster on-time output message 
