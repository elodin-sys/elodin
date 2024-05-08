Executive Summary
-----------------

This module filters position measurements that have been processed from planet images in order to estimate spacecraft relative position to an observed body in the inertial frame. The filter used is an unscented Kalman filter, and the images are first processed by houghCricles and pixelLineConverter in order to produce this filter's measurements.

The module
:download:`PDF Description </../../src/fswAlgorithms/opticalNavigation/relativeODuKF/_Documentation/Basilisk-relativeOD-20190620.pdf>`
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
    * - navStateOutMsg
      - :ref:`NavTransMsgPayload`
      - navigation translation output message
    * - filtDataOutMsg
      - :ref:`OpNavFilterMsgPayload`
      - output filter data message
    * - opNavInMsg
      - :ref:`OpNavMsgPayload`
      - opnav input message


