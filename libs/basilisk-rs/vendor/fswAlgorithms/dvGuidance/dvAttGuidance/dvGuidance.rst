Executive Summary
-----------------

This module creates a time varying attitude reference frame message that allows the orbit correction burn direction to rotate at a constant rate.

A message is read in containing the base \f$\Delta\mathbf{v}\f$ direction, the burn duration, as well as a nominal rotation axis.  A base burn frame is created relative to which a constant rotation about the 3rd frame axis is performed.  The output message contains the full reference frame states including the constant angular velocity vector and a zero angular acceleration vector. More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/dvGuidance/dvAttGuidance/_Documentation/Basilisk-dvGuidance-2019-03-28.pdf>`.

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
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference output message
    * - burnDataInMsg
      - :ref:`DvBurnCmdMsgPayload`
      - Input message that configures the vehicle burn
