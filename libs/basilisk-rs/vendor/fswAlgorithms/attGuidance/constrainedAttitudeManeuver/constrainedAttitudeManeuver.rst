Executive Summary
-----------------

Module to maneuver a spacecraft subject to hard rotational constraints. This module enables the user to add keep-in and keep-out zones that must not be violated
by certain user-specified body-frame directions during the maneuver. See the Detailed Module Description for a better description on how this is achieved.
The module produces an Attitude Reference Message containing the reference trajectory that is tracked by the spacecraft.


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
    * - attRefOutMsgC
      - :ref:`AttRefMsgPayload`
      - C-wrapped Output Attitude Reference Message.
    * - scStateInMsg
      - :ref:`SCStatesMsgPayload`
      - Input SC States Message.
    * - vehicleConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - Input Vehicle Configuration Message.
    * - keepOutCelBodyInMsg
      - :ref:`SpicePlanetStateMsgPayload`
      - Input keep-out Planet State Message.
    * - keepInCelBodyInMsg
      - :ref:`SpicePlanetStateMsgPayload`
      - Input keep-in Planet State Message.


Module Assumptions and Limitations
----------------------------------
The module, in its current form, allows to only include one keep-out and one keep-in celestial object(s). Multiple boresights (up to 10) can be specified as either
keep-in or keep-out body directions. The keep-out constraint is considered respected when all the keep-out body-fixed boresights are outside of the keep-out zone
simultaneously. On the contrary, the keep-in constraint is considered respected when at least one keep-in body-fixed direction is inside the keep-in zone.

At this stage of development, constraint compliance is only guaranteed for the attitude waypoints that are used as a base for the interpolation that yields the attitude
reference trajectory (see Detailed Module Description below). It may happen that portions of the reference trajectory between waypoints still violate the rotational
constraints. Ensuring that this does not happen will be subject of further refinements of this module. For the time being, it is possible to circumvent this problem
changing the grid refinement level ``N``. This changes the coordinates of the grid points, thus yielding a different reference trajectory for every ``N``. A small ``N``
results in a coarser grid, which is more likely to yield a trajectory that violates some constraints, but is generally smoother. Viceversa, a higher ``N`` gives a 
finer grid where the chance of obtaining a constraint-incompliant grid is reduced. However, the trajectory in this case is less regular due to the fact that the
interpolating curve is forced to pass through a larger number of waypoints. A higher ``N`` is also associated with a higher computational cost.


Detailed Module Description
---------------------------
A detailed explanation of the method implemented in this module can be found in `R. Calaon and H. Schaub, "Constrained Attitude Maneuvering via Modified-Rodrigues-Parameter-Based
Motion Planning Algorithms" <https://arc.aiaa.org/doi/abs/10.2514/1.A35294>`__. To summarize, this module builds a 3D grid in MRP space, whose density is regulated by the input
parameter ``N``. Each grid node corresponds to an attitude. An undirected graph is built with all the nodes that are constraint compliant. Two nodes are added to the graph:

- ``startNode`` whose coordinates :math:`\sigma_{BN,S}` correspond to the attitude of the spacecraft contained in the ``vehicleConfigInMsg``;
- ``goalNode`` whose coordinates :math:`\sigma_{BN,G}` correspond to the desired target attitude at the end of the maneuver.

Two different cost functions are used by the :math:`A^*` algorithm to search a valid path. The first is based on the total cartesian length of the path in MRP space. 
The second is the effort-based cost function computed integrating the control torque norm over the interpolated trajectory obtained from a path., as explained in
`R. Calaon and H. Schaub <https://arc.aiaa.org/doi/abs/10.2514/1.A35294>`__. In both cases, the final reference passed to the Attitude Reference Message
consists in the interpolated curve obtained from the optimal path computed by :math:`A^*`, based on the chosen cost function. Interpolation is performed using the 
routine in :ref:`BSpline`.

Note that this module does not implement the constant angular rate norm routine described in `R. Calaon and H. Schaub <https://arc.aiaa.org/doi/abs/10.2514/1.A35294>`__.
The attitude, rates and accelerations provided to the Attitude Reference Message are those obtained directly from the BSpline interpolation.


User Guide
----------
The required module configuration is::

    CAM = constrainedAttitudeManeuver.ConstrainedAttitudeManeuver(N)
    CAM.ModelTag = "constrainedAttitudeManeuvering"
    CAM.sigma_BN_goal = sigma_BN_G
    CAM.omega_BN_B_goal = [0, 0, 0]
    CAM.avgOmega = 0.04
    CAM.BSplineType = 0
    CAM.costFcnType = 0
    CAM.appendKeepOutDirection([1,0,0], keepOutFov)
    CAM.appendKeepInDirection([0,1,0], keepInFov)
    scSim.AddModelToTask(simTaskName, CAM)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 34 66
   :header-rows: 1

   * - Parameter
     - Description
   * - ``sigma_BN_goal``
     - goal MRP attitude set
   * - ``omega_BN_B_goal``
     - desired angular rate at goal, in body frame coordinates
   * - ``avgOmega``
     - average angular rate norm desired for the maneuver
   * - ``BSplineType``
     - desired type of BSpline: 0 for precise interpolation, 1 for least-squares approximation
   * - ``costFcnType``
     - desired cost function for the graph search algorithm: 0 for total MRP distance, 1 for effort-based cost.