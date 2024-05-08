Executive Summary
-----------------
This module implements and tests a Switch Unscented Kalman Filter in order to estimate an arbitrary heading direction.
More information on can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attDetermination/headingSuKF/_Documentation/heading_SuKF.pdf>`

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
    * - opnavDataOutMsg
      - :ref:`OpNavMsgPayload`
      - output message with opnav information
    * - filtDataOutMsg
      - :ref:`HeadingFilterMsgPayload`
      - output message with filter state data information
    * - opnavDataInMsg
      - :ref:`OpNavMsgPayload`
      - optical navigation input message
    * - cameraConfigInMsg
      - :ref:`CameraConfigMsgPayload`
      - (optional) camera configuration input message
