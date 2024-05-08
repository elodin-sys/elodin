Executive Summary
-----------------
This module computes a reference attitude frame that simultaneously satisfies multiple pointing constraints. The first constraint consists of aligning a body-frame direction :math:`{}^\mathcal{B}\hat{h}_1` with a certain inertial reference direction :math:`{}^\mathcal{N}\hat{h}_\text{ref}`. This locks two out of three degrees of freedom that characterize a rigid body rotation. The second constraints consists in achieving maximum power generation on the solar arrays, assuming that the solar arrays can rotate about their drive axis. This condition is obtained ensuring that the body-fixed solar array drive direction :math:`{}^\mathcal{B}\hat{a}_1` is as close to perpendicular as possible to the Sun direction. When maximum power generation is possible, two solutions can be found that satisfy the previous two constraints simultaneously. When this happens, it is possible to consider a third body-fixed direction :math:`{}^\mathcal{B}\hat{a}_2` which should remain as close as possible to the Sun direction, while maintaining the other two constraints satisfied. This allows to discriminate between the two frames, and to pick the one that drives :math:`{}^\mathcal{B}\hat{a}_2` closer to the Sun. It is possible to provide a second body frame direction :math:`{}^\mathcal{B}\hat{h}_2` as an optional parameter: in this case the module chooses whether to align :math:`{}^\mathcal{B}\hat{h}_1` or :math:`{}^\mathcal{B}\hat{h}_2` with :math:`{}^\mathcal{N}\hat{h}_\text{ref}` depending on which provides a better alignment of :math:`{}^\mathcal{B}\hat{a}_2` with the Sun direction.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages. The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - attNavInMsg
      - :ref:`NavAttMsgPayload`
      - Input message containing current attitude and Sun direction in body-frame coordinates. Note that, for the Sun direction to appear in the message, the :ref:`SpicePlanetStateMsgPayload` must be provided as input msg to :ref:`simpleNav`, otherwise the Sun direction is zeroed by default.
    * - bodyHeadingInMsg
      - :ref:`BodyHeadingMsgPayload`
      - (optional) Input message containing the body-frame direction :math:`{}^\mathcal{B}\hat{h}`. Alternatively, the direction can be specified as input parameter ``h1Hat_B``. When this input msg is connected, the input parameter is neglected in favor of the input msg.
    * - inertialHeadingInMsg
      - :ref:`InertialHeadingMsgPayload`
      - (optional) Input message containing the inertial-frame direction :math:`{}^\mathcal{N}\hat{h}_\text{ref}`. Alternatively, the direction can be specified as input parameter ``hHat_N``. When this input msg is connected, the input parameter is neglected in favor of the input msg.
    * - ephemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - (optional) Input message containing the inertial position of a celestial object, whose direction with respect to the spacecraft serves as the inertial reference direction :math:`{}^\mathcal{N}\hat{h}_\text{ref}`. This input msg must be provided together with ``transNavInMsg`` to compute the relative position of the celestial object to the spacecraft. If both ``inertialHeadingInMsg`` and ``ephemerisInMsg`` are connected, the inertial reference direction :math:`{}^\mathcal{N}\hat{h}_\text{ref}` is computed according to ``inertialHeadingInMsg``.
    * - transNavInMsg
      - :ref:`NavTransMsgPayload`
      - (optional) Input message containing the inertial position and velocity of the spacecraft. This message must be connected together with ``ephemerisInMsg`` to allow to compute :math:`{}^\mathcal{N}\hat{h}_\text{ref}`.
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - Output attitude reference message containing reference attitude, reference angular rates and accelerations.


Detailed Module Description
---------------------------
A detailed mathematical derivation of the equations applied by this module can be found in R. Calaon, C. Allard and H. Schaub, "Attitude Reference Generation for Spacecraft with Rotating Solar Arrays and Pointing Constraints", in preparation for Journal of Spacecraft and Rockets.
The input parameter ``alignmentPriority`` allows to choose whether the first or the second constraint is strictly enforced. When ``alignmentPriority = 0``, the body heading :math:`{}^\mathcal{B}\hat{h}` and the inertial heading :math:`{}^\mathcal{N}\hat{h}_\text{ref}` match exactly, while the incidence angle on the solar arrays is as close to optimal as possible. On the contrary, when ``alignmentPriority = 1``, the solar array drive :math:`{}^\mathcal{B}\hat{a}_1` is perpendicular to the Sun direction, to ensure maximum power generation, while the body heading and the inertial heading are as close to parallel as possible.

Attention must be paid to how these pieces of input information is provided:

  - Input body-frame heading: this can be specified either via the input parameter ``h1Hat_B``, or connecting the input message ``bodyHeadingInMsg``. Specifying the body-frame heading via the input parameter is desirable when such direction does not change over time; vice versa, when the body-frame heading is time varying, this needs to be passed via the ``bodyHeadingInMsg``. When both ``h1Hat_B`` and ``bodyHeadingInMsg`` are provided, the module ignores ``h1Hat_B`` and reads the body-frame direction from the input message.
  - Input inertial-frame heading: this can be specified via the input parameter ``hHat_N``, connecting the message ``inertialHeadingInMsg``, or connecting both the messages ``ephemerisInMsg`` and ``transNavInMsg``. The input parameter ``hHat_N`` is desirable when the inertial heading is fixed in time. The message ``inertialHeadingInMsg`` is needed when the heading direction is time-varying. Finally, providing ``ephemerisInMsg`` and ``transNavInMsg`` allows to compute the inertial heading as the vector difference between the inertial position of a celestial object and the position of the spacecraft: this is useful when the spacecraft needs to point a body-frame heading towards a celestial object. When all of these input messages are connected, the inertial heading is computed from the ``inertialHeadingInMsg``.

Module Assumptions and Limitations
----------------------------------
The limitations of this module are inherent to the geometry of the problem, which determines whether or not all the constraints can be satisfied. For example, as shown in  in R. Calaon, C. Allard and H. Schaub, "Attitude Reference Generation for Spacecraft with Rotating Solar Arrays and Pointing Constraints," In preparation for Journal of Spacecraft and Rockets, depending on the relative orientation of :math:`{}^\mathcal{B}h` and :math:`{}^\mathcal{B}a_1`, it may not be possible to  achieve perfect incidence angle on the solar arrays. Only when perfect incidence is obtained, it is possible to solve for the solution that also drives the body-fixed direction :math:`{}^\mathcal{B}a_2` close to the Sun. When perfect incidence is achievable, two solutions exist. If :math:`{}^\mathcal{B}a_2` is provided as input, this is used to determine which solution to pick. If this input is not provided, one of the two solution is chosen arbitrarily.

Due to the difficulty in developing an analytical formulation for the reference angular rate and angular acceleration vectors, these are computed via second-order finite differences. At every time step, the current reference attitude and time stamp are stored in a module variable and used in the following time updates to compute angular rates and accelerations via finite differences.


User Guide
----------
The required module configuration is::

    attReference = oneAxisSolarArrayPoint.oneAxisSolarArrayPoint()
    attReference.ModelTag = "threeAxesPoint"
    attReference.a1Hat_B = a1_B
    attReference.alignmentPriority = 0
    scSim.AddModelToTaskAddModelToTask(simTaskName, attReference)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``a1Hat_B``
     - [0, 0, 0]
     - solar array drive direction, it must be specified by the user
   * - ``alignmentPriority``
     - 0
     - 0 to prioritize first constraint, 1 to prioritize second constraint
   * - ``h1Hat_B`` (optional)
     - [0, 0, 0]
     - body-frame heading
   * - ``hHat_N`` (optional)
     - [0, 0, 0]
     - inertial-frame heading
   * - ``a2Hat_B`` (optional)
     - [0, 0, 0]
     - third body frame direction that should be as close as possible to Sun direction.
   * - ``h2Hat_B`` (optional)
     - [0, 0, 0]
     - second body-frame heading