Executive Summary
-----------------


This method applies a two-level deadbanding logic (according to the current average simple compared with the set threshold)
and decides whether control should be switched ON/OFF or not.

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
    * - attGuidOutMsg
      - :ref:`AttGuidMsgPayload`
      - attitude guidance output message
    * - guidInMsg
      - :ref:`AttGuidMsgPayload`
      - incoming attitude guidance message

