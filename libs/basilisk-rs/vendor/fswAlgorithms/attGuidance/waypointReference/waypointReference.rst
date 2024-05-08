Executive Summary
-----------------

Module that reads the reorientation maneuver of a spacecraft from a text file, likely created outside of Basilisk, and outputs an 
Attitude Reference Message. This module makes it possible to reproduce on Basilisk attitude orientation maneuvers computed externally. The module outputs an Attitude
Reference Message that follows the sequence of waypoints contained in the text file. The text file must be formatted appropriately for the module to be able to read 
the information correctly: see Module Assumptions and Limitaions for a detailed explanation on how to do this.


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
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - Output Attitude Reference Message.


Module Assumptions and Limitations
----------------------------------
The module assumes that the text file is written in a compatible form, which means that each piece of information must be provided in the correct order.
Each line of text should contain information relative to one and only one waypoint along the maneuver: such information must be in the following order: time, 
attitude parameters, angular rates, angular acceleration. There must be a ``delimiter`` between each piece of information and the next one, as well al between 
different elements of the same piece of information (for example between consecutive entries of the 3-dimensional angular rate vector).  Using MRP to represent
the attitude :math:`\sigma_{\mathcal{R/N}}=[\sigma_1, \sigma_2, \sigma_3]`, expressing angular rates :math:`{}^{\mathcal{N}}\omega_{\mathcal{R/N}}=[\omega_1, \omega_2, \omega_3]` 
and accelerations :math:`{}^{\mathcal{N}}\dot{\omega}_{\mathcal{R/N}}=[\dot{\omega}_1, \dot{\omega}_2, \dot{\omega}_3]` in the inertial frame, and using the comma as a delimiter, 
one waypoint is correctly read if presented as a line like the following

.. math::
    t, \sigma_1, \sigma_2, \sigma_3, \omega_1, \omega_2, \omega_3, \dot{\omega}_1, \dot{\omega}_2, \dot{\omega}_3
	
The module is conceptually very simple and makes no further assumptions. However, the user might want to use a sampling frequency in the
Basilisk simulation that is equal or higher than the frequency of the waypoints. For lower sampling frequencies, the module output does not give a 
trustworthy representation of the maneuver.



Detailed Module Description
---------------------------
The module reads a sequence of time-tagged waypoints. Defining :math:`t=[t_0,...,t_N]` the times of the N+1 waypoints, and :math:`t_{sim}` the simulation time, we have that:

- for :math:`t_{sim} < t_0`: the attitude is held constant and equal to the attitude of the first waypoint; angular rates and acceleration are kept at zero;
- for :math:`t_0 \leq t_{sim} \leq t_N`: attitude, angular rates and accelerations are the result of linear interpolation between the closest two waypoints;
- for :math:`t_{sim} > t_N`: the attitude is held constant and equal to the attitude of the last waypoint; angular rates and acceleration are kept at zero.

When reading from the data file, the module always maps the attitude to the short rotation MRP set, regardless of the attitude type. This means that, for a data file that
describes large attitude rotations (larger than 180 deg), the ``attRefOutMsg.sigma_RN`` will present a discontinuity. When two subsequent waypoints are mapped into different 
MRP sets, the interpolation is carried out in that time interval between the first waypoint and the shadow set of the second waypoint. This allows for a non-singular attitude
description.
		
		
User Guide
----------
The module assumes the data file is in plain text form and the following format:

- time (seconds)
- attitude parameters (MRPs or EPs)
- angular rates (rad/s) either expressed in inertial frame or reference frame
- angular accelerations (rad/s^2) either expressed in inertial frame or reference frame

where each line contains information about only one intermediate point of the maneuver.


The required module configuration is::

    waypointReferenceModule = waypointReference.WaypointReference()
    waypointReferenceModule.ModelTag = "waypointReference"
    waypointReferenceModule.dataFileName = dataFileName
    waypointReferenceModule.attitudeType = 0
    unitTestSim.AddModelToTask(unitTaskName, waypointReferenceModule)
	
Note that for ``attitudeType``, a valid input must be provided by the user: 0 - MRP, 1 - EP or quaternions (q0, q1, q2, q3), 2 - EP or quaternions (q1, q2, q3, qs).
No default attitude type is used by the module, therefore faliure to specify this parameter results in breaking the simulation.

The module is configurable with the following optional parameters:

.. list-table:: Module Optional Parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``delimiter``
     - ","
     - delimiter string that separates data on a line
   * - ``useReferenceFrame``
     - false
     - if true, reads angular rates and accelerations in the reference frame instead of inertial frame
   * - ``headerLines``
     - 0
     - number of header lines in the data file that should be ignored before starting to read in the waypoints
