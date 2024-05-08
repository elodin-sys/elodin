Executive Summary
-----------------

Converter that takes a image processing message and camera information and outputs a relative position to the object.

The module
:download:`PDF Description </../../src/fswAlgorithms/imageProcessing/pixelLineConverter/_Documentation/Basilisk-pixelLineConverter-20190524.pdf>`
contains further information on this module's function,
how to run it, as well as testing.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_pixelLineConverter:
.. figure:: /../../src/fswAlgorithms/imageProcessing/pixelLineConverter/_Documentation/Images/moduleImgPixelLineConverter.svg
    :align: center

    Figure 1: ``pixelLineConverter()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - opNavOutMsg
      - :ref:`OpNavMsgPayload`
      - optical navigation output message
    * - cameraConfigInMsg
      - :ref:`CameraConfigMsgPayload`
      - camera config input message
    * - attInMsg
      - :ref:`NavAttMsgPayload`
      - attitude input message
    * - circlesInMsg
      - :ref:`OpNavCirclesMsgPayload`
      - circles input message

