Executive Summary
-----------------

This module takes the star tracker sensor data in the platform frame and converts that information to the format used by the ST nav in the body frame.

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
    * - stSensorInMsg
      - :ref:`STSensorMsgPayload`
      - star tracker sensor input message
    * - stAttOutMsg
      - :ref:`STAttMsgPayload`
      - star tracker attitude output message

