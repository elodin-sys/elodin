Executive Summary
-----------------


This module is responsible for correcting the raw CSS output values to the expected cosine values. This requires a pre-calibrated Chebyshev residual model which calculates the expected deviation from the expected CSS cosine output given a raw CSS measurement at a given distance from the sun. More information on can be found in the
:download:`PDF Description </../../src/fswAlgorithms/sensorInterfaces/CSSSensorData/_Documentation/Basilisk-CSSSensorDataModule-20190207.pdf>`.

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
    * - sensorListInMsg
      - :ref:`CSSArraySensorMsgPayload`
      - input message that contains CSS data
    * - cssArrayOutMsg
      - :ref:`CSSArraySensorMsgPayload`
      - output message of corrected CSS data

