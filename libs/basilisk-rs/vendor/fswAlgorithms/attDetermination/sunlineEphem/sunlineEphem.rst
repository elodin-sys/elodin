Executive Summary
-----------------

This module computes an ephemeris-based sunline heading.

More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attDetermination/sunlineEphem/_Documentation/Basilisk-SunlineEphem-20181204.pdf>`

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
    * - navStateOutMsg
      - :ref:`NavAttMsgPayload`
      - name of the navigation output message containing the estimated states
    * - sunPositionInMsg
      - :ref:`EphemerisMsgPayload`
      - name of the sun ephemeris input message
    * - scPositionInMsg
      - :ref:`NavTransMsgPayload`
      - name of the spacecraft ephemeris input message
    * - scAttitudeInMsg
      - :ref:`NavAttMsgPayload`
      - name of the spacecraft attitude input message

