Executive Summary
-----------------

This module creates a dynamic reference frame attitude state message where the initial orientation relative to the input reference frame is specified through an MRP set, and the angular velocity vector is held fixed as seen by the resulting reference frame. More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/mrpRotation/_Documentation/Basilisk-MRPROTATION-20180522.pdf>`.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_mrpRotation:
.. figure:: /../../src/fswAlgorithms/attGuidance/mrpRotation/_Documentation/Images/moduleIOMrpRotation.svg
    :align: center

    Figure 1: ``mrpRotation()`` Module I/O Illustration


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

