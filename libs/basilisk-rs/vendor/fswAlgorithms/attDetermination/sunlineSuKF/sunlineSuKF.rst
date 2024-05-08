Executive Summary
-----------------

This module implements and tests a Switch Unscented Kalman Filter in order to estimate the sunline direction. The theory used in this module can be found in this `paper <https://www.researchgate.net/publication/3908304_The_Square-Root_Unscented_Kalman_Filter_for_State_and_Parameter-Estimation>`__.

More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attDetermination/sunlineSuKF/_Documentation/Sunline_SuKF.pdf>`


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
    * - filtDataOutMsg
      - :ref:`SunlineFilterMsgPayload`
      - name of the output filter data message
    * - cssDataInMsg
      - :ref:`CSSArraySensorMsgPayload`
      - name of the CSS sensor input message
    * - cssConfigInMsg
      - :ref:`CSSConfigMsgPayload`
      - name of the CSS configuration input message

