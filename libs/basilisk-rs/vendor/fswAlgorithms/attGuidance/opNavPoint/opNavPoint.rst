Executive Summary
-----------------

This module implements a opNav point attitude guidance routine.

This algorithm is intended to be incredibly simple and robust: it finds the angle error between the camera boresight (or desired control axis in the camera frame) and the planet heading in the camera frame and brings them to zero. This is analoguous to sunSafePoint.  The file
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/opNavPoint/_Documentation/Basilisk-opNavPoint-20190820.pdf>`.
contains further information on this module's function, how to run it, as well as testing.

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
    * - attGuidanceOutMsg
      - :ref:`AttGuidMsgPayload`
      - name of the output guidance message
    * - opnavDataInMsg
      - :ref:`OpNavMsgPayload`
      - name of the optical navigation input message
    * - imuInMsg
      - :ref:`NavAttMsgPayload`
      - name of the incoming IMU message
    * - cameraConfigInMsg
      - :ref:`CameraConfigMsgPayload`
      - name of the camera config message

