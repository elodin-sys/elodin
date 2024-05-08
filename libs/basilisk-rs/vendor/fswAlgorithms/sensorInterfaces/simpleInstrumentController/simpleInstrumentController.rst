Executive Summary
-----------------
This module generates a command in the form of a :ref:`DeviceCmdMsgPayload` that turns on a :ref:`simpleInstrument`
if the spacecraft a.) has access to a :ref:`groundLocation` and b.) the associated attitude error and attitude rate
error from an attitude guidance message is within the given tolerance.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.
The module msg connection is set by the user from python.
The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - locationAccessInMsg
      - :ref:`AccessMsgPayload`
      - input msg containing the ground location access
    * - attGuidInMsg
      - :ref:`AttGuidMsgPayload`
      - input message containing the attitude guidance
    * - deviceStatusInMsg
      - :ref:`DeviceStatusMsgPayload`
      - (optional) input message containing the device status
    * - deviceCmdOutMsg
      - :ref:`DeviceCmdMsgPayload`
      - output message with the device command

Detailed Module Description
---------------------------
This module writes out a :ref:`DeviceCmdMsgPayload` to turn on an instrument, i.e. :ref:`simpleInstrument`.

.. note::

    This module assumes that the simulated data is generated instantaneously. Therefore, the ``baudRate`` of the
    simpleInstrument must be sized as such.

The module first checks if the instrument controller is active using the ``controllerStatus`` variable. This 
variable defaults to 1 (active), but the user may set the variable to 0 (inactive) with the ``deviceStatusInMsg`` 
or ``controllerStatus`` variable. Then, the module checks if target has already been imaged.  If the target has 
not been imaged, the module then checks if the norm of the attitude error is less than the user specified tolerance 
and if the :ref:`groundLocation` is accessible. If both are true, the module sets the ``deviceCmd`` to 1. Otherwise, 
the ``deviceCmd`` is set to 0.

User Guide
----------
Two variables must be set. The variable ``attErrTolerance`` is the norm of the acceptable attitude error in MRPs.
It must be set at the beginning of the sim.

The ``imaged`` variable is always initialized to 0 (i.e. the target has not been imaged). However, if the simulation
is stopped and restarted again this variable should be reinitialized to 0 in between. If it is not and the previous
target was imaged, the new target will not be imaged.

Optionally, attitude rate error checking may be enabled by setting ``useRateTolerance`` to ``1``. If enabled,
``rateErrTolerance`` should be set to the norm of the acceptable attitude rate error in rad/s when imaging.

If the user desires to make the instrument controller inactive, the ``controllerStatus`` variable should be set to 0. 
This may be done if the controller is to match the status of corresponding instrument data and power modules. This
may also be accomplished by writing out a :ref:`DeviceStatusMsgPayload` to the ``deviceStatusInMsg``.