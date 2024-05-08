Executive Summary
-----------------

This attitude guidance module computes the velocity reference frame states.

The orbit can be any type of Keplerian motion, including circular, elliptical or hyperbolic.  The module
:download:`PDF Description </../../src/fswAlgorithms/attGuidance/velocityPoint/_Documentation/velocityPoint.pdf>`.
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
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference output message
    * - transNavInMsg
      - :ref:`NavTransMsgPayload`
      - incoming spacecraft translational state message
    * - celBodyInMsg
      - :ref:`EphemerisMsgPayload`
      - (optional )primary celestial body information input message

