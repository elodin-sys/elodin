Executive Summary
-----------------

Converter that takes a limb message and camera information and outputs a relative position to the object. This algorithm was developed by J. Christian.

The module
:download:`PDF Description </../../src/fswAlgorithms/imageProcessing/horizonOpNav/_Documentation/Basilisk-horizonOpNav-20190918.pdf>`
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
    * - opNavOutMsg
      - :ref:`OpNavMsgPayload`
      - output navigation message for relative position
    * - cameraConfigInMsg
      - :ref:`CameraConfigMsgPayload`
      - camera config input message
    * - attInMsg
      - :ref:`NavAttMsgPayload`
      - attitude input message
    * - limbInMsg
      - :ref:`OpNavLimbMsgPayload`
      - limb input message




