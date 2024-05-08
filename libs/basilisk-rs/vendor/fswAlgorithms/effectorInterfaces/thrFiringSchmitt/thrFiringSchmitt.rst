Executive Summary
-----------------

A Schmitt trigger logic is implemented to map a desired thruster force value into a thruster on command time.

The module reads in the attitude control thruster force values for both on- and off-pulsing scenarios, and then maps this into a time which specifies how long a thruster should be on.  The thruster configuration data is read in through a separate input message in the reset method.  The Schmitt trigger allows for an upper and lower bound where the thruster is either turned on or off. More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/effectorInterfaces/thrFiringSchmitt/_Documentation/Basilisk-thrFiringSchmitt-2019-03-29.pdf>`.
The paper `Steady-State Attitude and Control Effort Sensitivity Analysis of Discretized Thruster Implementations <https://doi.org/10.2514/1.A33709>`__ includes a detailed discussion on the Schmitt Trigger and compares it to other thruster firing methods.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_thrFiringSchmitt:
.. figure:: /../../src/fswAlgorithms/effectorInterfaces/thrFiringSchmitt/_Documentation/Images/moduleImgThrFiringSchmitt.svg
    :align: center

    Figure 1: ``thrFiringSchmitt()`` Module I/O Illustration


.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - thrForceInMsg
      - :ref:`THRArrayCmdForceMsgPayload`
      - thruster force input message
    * - onTimeOutMsg
      - :ref:`THRArrayOnTimeCmdMsgPayload`
      - thruster on-time output message
    * - thrConfInMsg
      - :ref:`THRArrayConfigMsgPayload`
      - Thruster array configuration input message

