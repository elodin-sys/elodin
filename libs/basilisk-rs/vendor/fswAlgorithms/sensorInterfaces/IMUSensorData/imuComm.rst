Executive Summary
-----------------

Converts incoming IMU data in the sensor platform frame P to the spacecraft body frame B.

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
    * - imuComInMsg
      - :ref:`IMUSensorMsgPayload`
      - imu input message in sensor platform frame
    * - imuSensorOutMsg
      - :ref:`IMUSensorBodyMsgPayload`
      - imu output message in spacecraft body frame


User Guide
----------
The only variable that must be set is the DCM from the platform frame P to the body frame B, ``dcm_BP``.

