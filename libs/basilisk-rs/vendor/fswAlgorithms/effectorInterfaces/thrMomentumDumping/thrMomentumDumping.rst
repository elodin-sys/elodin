Executive Summary
-----------------

This module reads in the desired impulse that each thruster must produce to create inertial momentum change to despin the RWs.

The output of the module is a setup of thruster firing times.  Each thruster can only fire for a maximum time that matches a single control period.  After this the thrusters are off for an integer number of control periods to let the RW re-stabilize the attitude about an inertial pointing scenario. The module
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/thrMomentumDumping/_Documentation/Basilisk-thrMomentumDumping-20160820.pdf>` contains further information on this module's function, how to run it, as well as testing.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_thrMomentumDumping:
.. figure:: /../../src/fswAlgorithms/effectorInterfaces/thrMomentumDumping/_Documentation/Images/moduleImgThrMomentumDumping.svg
    :align: center

    Figure 1: ``thrMomentumDumping()`` Module I/O Illustration


.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - thrusterOnTimeOutMsg
      - :ref:`THRArrayOnTimeCmdMsgPayload`
      - thruster on time output message
    * - thrusterImpulseInMsg
      - :ref:`THRArrayCmdForceMsgPayload`
      - commanded thruster impulse input message
    * - thrusterConfInMsg
      - :ref:`THRArrayConfigMsgPayload`
      - Thruster array configuration input message
    * - deltaHInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - requested momentum change input message

