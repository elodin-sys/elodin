Executive Summary
-----------------
This module reads in a message of type :ref:`TAMSensorBodyMsgPayload`, outputs the magnetometer measurement vector in vehicle's body coordinates ``tam_B`` to the output message ``tamOutMsg``.

Module Assumptions and Limitations
----------------------------------
No assumptions are made.

Message Connection Descriptions
-------------------------------
The following table lists the module input and output messages.  The module msg variable name is set by the user from python.  The msg type contains a link to the message structure definition, while the description provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - tamInMsg
      - :ref:`TAMSensorMsgPayload`
      - TAM sensor interface input message
    * - tamOutMsg
      - :ref:`TAMSensorBodyMsgPayload`
      - TAM sensor interface output message



User Guide
----------
In order to transform the ``tam_S`` vector of :ref:`TAMSensorMsgPayload` from sensor to body frame, ``dcm_BS`` should be defined.
