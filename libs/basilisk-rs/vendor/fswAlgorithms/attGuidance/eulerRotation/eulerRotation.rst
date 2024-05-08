Executive Summary
-----------------

This module guidance modules creates constant Euler angle rate rotations about a primary axis
to create dynamic reference frames.

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
      - name of the output message containing the Reference
    * - attRefInMsg
      - :ref:`AttRefMsgPayload`
      - name of the guidance reference input message
    * - desiredAttInMsg
      - :ref:`AttStateMsgPayload`
      - (optional) name of the incoming message containing the desired Euler angle set

User Guide
----------
The initial orientation of the dynamic reference frame is set through the module variable ``angleSet``.  This is a
3-2-1 Euler angle sequence.

To set the desired constant 3-2-1 Euler angel rates, set the module variable ``angleRates``.
