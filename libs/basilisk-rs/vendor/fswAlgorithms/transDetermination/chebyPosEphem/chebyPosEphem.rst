Executive Summary
-----------------

This module allows the user to specify a set of Chebyshev
coefficients to fit a space ephemeris trajectory. Next the input time is used to determine where a given body is in space.

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
    * - posFitOutMsg
      - :ref:`EphemerisMsgPayload`
      - output navigation message for pos/vel
    * - clockCorrInMsg
      - :ref:`TDBVehicleClockCorrelationMsgPayload`
      - clock correlation input message
