Executive Summary
-----------------

The primary purpose of this module is to provide an attitude reference output to make sure that a vector given in the deputy spacecraft's body frame points to the chief spacecraft. The position of the chief- and deputy spacecraft in the inertial frame are used as inputs for this module. The module uses the positions to create a reference vector that points from the deputy to the chief. A coordinate system is built around this vector and the orientation, angular velocity, and angular acceleration of this coordinate system are calculated with respect to the inertial frame. The output consists of these three vectors and can consequently be used as an input for the attitude tracking error module.

More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/formationFlying/spacecraftPointing/_Documentation/Basilisk-spacecraftPointing-20190116.pdf>`.

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
    * - chiefPositionInMsg
      - :ref:`NavTransMsgPayload`
      - primary position input message
    * - deputyPositionInMsg
      - :ref:`NavTransMsgPayload`
      - secondary position input message
    * - attReferenceOutMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference output message

