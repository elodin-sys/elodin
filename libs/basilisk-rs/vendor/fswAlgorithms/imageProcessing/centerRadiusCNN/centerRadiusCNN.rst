Executive Summary
-----------------
This module implements any convolutional neural for image processing. More precisely, the module uploads a trained model
and reads it using the OpenCV library. This module is then used on an image in order to extract a radius and center.


Module Assumptions and Limitations
----------------------------------
The module's assumptions are limited to the model it uploads. The training and performance of this module is not
protected by this implementation. This assumption is seen in the pixelNoise variable where the user sets the
performance of the net. 

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the user from python.  The msg type contains a link to the message structure definition, while the description provides information on what this message is used for.



.. table:: Module I/O Messages
    :widths: 25 25 100

    +-----------------------+---------------------------------+---------------------------------------------------+
    | Msg Variable Name     | Msg Type                        | Description                                       |
    +=======================+=================================+===================================================+
    | imageInMsg            | :ref:`CameraImageMsgPayload`    | (optional) Input image message.                   |
    |                       |                                 | This message either comes from the camera module  |
    |                       |                                 | or the viz interface if no noise is added.        |
    +-----------------------+---------------------------------+---------------------------------------------------+
    | opnavCirclesOutMsg    | :ref:`OpNavCirclesMsgPayload`   | Circle found in the image.                        |
    +-----------------------+---------------------------------+---------------------------------------------------+


User Guide
----------

The module is set easily using the path to the module and message names:

.. code-block:: python
    :linenos:

    moduleConfig.pathToNetwork = path + "/../position_net2_trained_11-14.onnx"
    moduleConfig.pixelNoise = [5,5,5]

