Executive Summary
-----------------

This module filters incoming star tracker measurements and reaction wheel data in order to get the best possible inertial attitude estimate. The filter used is an unscented Kalman filter using the Modified Rodrigues Parameters (MRPs) as a non-singular attitude measure.  Measurements can be coming in from several camera heads.

More information on can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attDetermination/InertialUKF/_Documentation/Basilisk-inertialUKF-20190402.pdf>`

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
      - navigation output message
    * - filtDataOutMsg
      - :ref:`InertialFilterMsgPayload`
      - name of the output filter data message
    * - massPropsInMsg
      - :ref:`VehicleConfigMsgPayload`
      - spacecraft vehicle configuration input message
    * - rwParamsInMsg
      - :ref:`RWArrayConfigMsgPayload`
      - reaction wheel parameter input message.  Can be an empty message if no RW are included.
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - reaction wheel speed input message.  Can be an empty message if no RW are included.
    * - gyrBuffInMsg
      - :ref:`AccDataMsgPayload`
      - rate gyro input message

