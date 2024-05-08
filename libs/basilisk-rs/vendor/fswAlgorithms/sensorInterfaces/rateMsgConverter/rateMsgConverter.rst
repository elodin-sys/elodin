Executive Summary
-----------------

The purpose of this module is to read in the IMU sensor body message from message type :ref:`IMUSensorBodyMsgPayload`, and store it in the output message of type :ref:`NavAttMsgPayload`.  The output message of type  :ref:`NavAttMsgPayload` is used as a common attitude input message by the FSW modules.  This output message is first zeroed, and then the rate vector is copied into the variable `omega\_BN\_B`. 

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
    * - navRateOutMsg
      - :ref:`NavAttMsgPayload`
      - attitude output message
    * - imuRateInMsg
      - :ref:`IMUSensorBodyMsgPayload`
      - attitude input message



