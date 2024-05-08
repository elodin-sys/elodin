Executive Summary
-----------------
Module to perform continuous instrument control if attitude error and attitude rate are within specified 
tolerances. If within the specified tolerances, the module will send an imaging command to the instrument. 
Applicable use case could be a continuous nadir imaging (scanning), where it must constantly check for 
attitude requirements. 

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
    * - accessInMsg
      - :ref:`AccessMsgPayload`
      - Ground location access input message
    * - attGuidInMsg
      - :ref:`AttGuidMsgPayload`
      - Attitude guidance input message
    * - deviceStatusInMsg
      - :ref:`DeviceStatusMsgPayload`
      - (optional) Device status input message
    * - deviceCmdOutMsg
      - :ref:`DeviceCmdMsgPayload`
      - Device status command output message

Detailed Module Description
---------------------------
This module writes out a :ref:`DeviceCmdMsgPayload` to turn on an instrument, i.e. :ref:`simpleInstrument`, continuously 
while the requirement conditions are met.

The module first checks if the instrument controller is active using the ``controllerStatus`` variable. This variable 
defaults to 0 (inactive) due to swig initialization of the module, but the user may set the variable to 1 (active) 
with the ``deviceStatusInMsg`` or ``controllerStatus`` variable. If the instrument controller is active, the module 
then checks if the norm of the attitude error is less than the user specified tolerance and if the :ref:`groundLocation` 
is accessible. If both are true, the module sets the ``deviceCmd`` to 1. Otherwise, the ``deviceCmd`` is set to 0.

User Guide
----------
Two variables must be set, ``attErrTolerance`` and ``useRateTolerance``. The variable ``attErrTolerance`` is the norm 
of the acceptable attitude error in MRPs; It must be set at the beginning of the sim. Also, ``useRateTolerance`` must 
be set to specify if the module should check for attitude rate requirements (set ``useRateTolerance`` to 1 to check, 
otherwise set to 0). 

If ``useRateTolerance`` is set to 1, ``rateErrTolerance`` must be set to the norm of the acceptable attitude rate
in rad/s. Otherwise, if ``useRateTolerance`` is set to 0, ``rateErrTolerance`` is not used.

If the user desires to make the instrument controller inactive, the ``controllerStatus`` variable should be set to 0. 
This may be done if the controller is to match the status of corresponding instrument data and power modules. This
may also be accomplished by writing out a :ref:`DeviceStatusMsgPayload` to the ``deviceStatusInMsg``.