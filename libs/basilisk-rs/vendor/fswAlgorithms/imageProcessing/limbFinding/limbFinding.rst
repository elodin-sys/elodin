Executive Summary
-----------------

Module reads in a message containing a pointer to an image and writes out the pixels that are on the lit limb of the planet.

The module
:download:`PDF Description </../../src/fswAlgorithms/imageProcessing/limbFinding/_Documentation/Basilisk-limbFinding-20190916.pdf>`
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
    * - opnavLimbOutMsg
      - :ref:`OpNavLimbMsgPayload`
      - output navigation message for relative position
    * - imageInMsg
      - :ref:`CameraImageMsgPayload`
      - camera image input message



