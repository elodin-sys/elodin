Executive Summary
-----------------

The primary purpose of this module is to schedule burn timing and attitude for spacecraft formation reconfiguration.
Basic idea is described in section 14.8.3 of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.
Based on keplerian dynamics, formation reconfiguration from one initial orbital element difference to target orbital element difference
is scheduled so that reconfiguration is completed in one orbit period.

In addition to formation control algorithm described in the textbook, some extensions and improvements are included in
this module.
First, this module assumes that deputy spacecraft has one-axis thrusters. Therefore, attitude control is also necessary 
along with burn at certain period. When burn timing is approaching, target attitude is output as ``attRefOutMsg``.
Otherwise, and if ``attRefInMsg`` (which is optional) is set, the reference message is output as ``attRefOutMsg``.
Second, if :math:`\delta a` is not zero, drift of :math:`\delta M` occurs. This module can take this drift into consideration
. Therefore, this module can achieve formation reconfiguration in orbital period.
Third, in general three-time burn is necessary for reconfiguration. Two of them occur at perigee and apogee each.
The other burn timing varies depending on necessary change of orbital element difference.
In some cases, this burn timing can be close to perigee or apogee. In these cases, two burns are integrated into one burn.
Parameter attControlTime is used to check whether this integration is necessary or not.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_spacecraftReconfig:

.. table:: Module I/O Messages
    :widths: 25 25 100

    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | Msg Variable Name        | Msg Type                               | Description                                                   |
    +==========================+========================================+===============================================================+
    | chiefTransInMsg          | :ref:`NavTransMsgPayload`              | chief's position and velocity input message                   |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | deputyTransInMsg         | :ref:`NavTransMsgPayload`              | deputy's position and velocity input message                  |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | thrustConfigInMsg        | :ref:`THRArrayConfigMsgPayload`        | deputy's thruster configuration input message                 |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | attRefInMsg              | :ref:`AttRefMsgPayload`                | (optional) deputy's reference attitude                        |
    |                          |                                        | input message. If set, then the deputy will point along this  |
    |                          |                                        | reference attitude unless it must point the thrusters in a    |
    |                          |                                        | control direction.                                            |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | vehicleConfigInMsg       | :ref:`VehicleConfigMsgPayload`         | deputy's vehicle configuration input message                  |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | attRefOutMsg             | :ref:`AttRefMsgPayload`                | deputy's target attitude output message                       |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | onTimeOutMsg             | :ref:`THRArrayOnTimeCmdMsgPayload`     | The deputy's thruster's on time output message                |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+
    | burnArrayInfoOutMsg      | :ref:`ReconfigBurnArrayInfoMsgPayload` | deputy's scheduled burns info output message                  |
    +--------------------------+----------------------------------------+---------------------------------------------------------------+

Module Assumptions and Limitations
----------------------------------
- This module uses classic orbital element, so this module cannot be applied to near-circular or near-equatorial orbits.
- Too long or too short attControlTime may result in control error.
- Impulsive maneuvers are approximated by steady thrust of a certain period.

User Guide
----------------------------------
This module requires the following variables to be set as parameters:

- ``attControlTime`` time [s] necessary to control one attitude to another attitude
- ``mu`` gravitational constant for a central body in m^3/s^2
- ``targetClassicOED`` desired orbital element difference.

For ``targetClassicOED``, normalized semi major axis must be used.
