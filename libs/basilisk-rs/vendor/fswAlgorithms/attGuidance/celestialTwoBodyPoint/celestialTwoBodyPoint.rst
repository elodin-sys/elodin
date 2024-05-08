Executive Summary
-----------------

This module point one body-fixed axis towards a primary celestial object.  The secondary goal is to point a second body-fixed axis towards another celestial object.

For example, the goal is to point the sensor towards the center of a planet while doing the best to keep the solar panel normal point at the sun.The module
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/celestialTwoBodyPoint/_Documentation/Basilisk-celestialTwoBodyPoint-20190311.pdf>`
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
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference output message
    * - transNavInMsg
      - :ref:`NavTransMsgPayload`
      - spacecraft translation motion input message
    * - celBodyInMsg
      - :ref:`EphemerisMsgPayload`
      - primary celestial body information input message
    * - secCelBodyInMsg
      - :ref:`EphemerisMsgPayload`
      - (optional) secondary celestial body information

