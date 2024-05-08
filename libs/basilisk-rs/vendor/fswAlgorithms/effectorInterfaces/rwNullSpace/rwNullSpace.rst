Executive Summary
-----------------

This module uses the Reaction Wheel (RW) null space to slow down the wheels.  The resulting motor torques are super imposed on top of the attitude feedback control RW motor torques.  More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/rwNullSpace/_Documentation/Basilisk-rwNullSpace-20190209.pdf>`.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_NullSpace:
.. figure:: /../../src/fswAlgorithms/effectorInterfaces/rwNullSpace/_Documentation/Images/moduleImgNullSpace.svg
    :align: center

    Figure 1: ``rwNullSpace()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - rwMotorTorqueInMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - RW motor torque input message
    * - rwSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - RW speed message
    * - rwDesiredSpeedsInMsg
      - :ref:`RWSpeedMsgPayload`
      - (optional) input message with the desired RW speeds
    * - rwConfigInMsg
      - :ref:`RWConstellationMsgPayload`
      - RW constellation configuration input message
    * - rwMotorTorqueOutMsg
      - :ref:`ArrayMotorTorqueMsgPayload`
      - RW motor torque output message





