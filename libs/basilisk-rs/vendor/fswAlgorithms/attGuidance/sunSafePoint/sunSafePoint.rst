Executive Summary
-----------------

This module provides the attitude guidance output for a sun pointing mode.  This could be used for safe mode, or a power generation mode.  The input is the sun direction vector which does not have to be normalized, as well as the body rate information.  The output is the standard BSK attitude reference state message.  The sun direction measurement is cross with the desired body axis that is to point at the sun to create a principle rotation vector.  The dot product between these two vectors is used to extract the principal rotation angle.  With these a tracking error MRP state is computer.  The body rate tracking errors relative to the reference frame are set equal to the measured body rates to bring the vehicle to rest when pointing at the sun.  Thus, the reference angular rate and acceleration vectors relative to the inertial frame are nominally set to zero.  If the sun vector is not available, then the reference rate is set to a body-fixed value while the attitude tracking error is set to zero.

The file
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/sunSafePoint/_Documentation/Basilisk-sunSafePoint-20180427.pdf>`.
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
      - attitude guidance output message
    * - sunDirectionInMsg
      - :ref:`NavAttMsgPayload`
      - sun direction input message
    * - imuInMsg
      - :ref:`NavAttMsgPayload`
      - IMU input message

