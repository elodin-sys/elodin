Executive Summary
-----------------
This module is provides a feedback control law for waypoint-to-waypoint control about a small body. The waypoints are defined in the Hill frame of the body.

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
    * - navTransInMsg
      - :ref:`NavTransMsgPayload`
      - translational navigation input message
    * - navAttInMsg
      - :ref:`NavAttMsgPayload`
      - attitude navigation input message
    * - asteroidEphemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - asteroid ephemeris input message
    * - sunEphemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - sun ephemeris input message
    * - forceOutMsg
      - :ref:`CmdForceBodyMsgPayload`
      - force command output
    * - forceOutMsgC
      - :ref:`CmdForceBodyMsgPayload`
      - C-wrapped force output message

Detailed Module Description
---------------------------
General Function
^^^^^^^^^^^^^^^^
The ``smallBodyWaypointFeedback()`` module provides a solution for waypoint-to-waypoint control about a small body. The feedback
control law is similar to the cartesian coordinate continuous feedback control law in Chapter 14 of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.
A cannonball SRP model, third body perturbations from the sun, and point-mass gravity are utilized. The state inputs are
messages written out by :ref:`simpleNav` and :ref:`planetNav` modules or an estimator that provides the same input messages.

Algorithm
^^^^^^^^^^
The state vector is defined as follows:

.. math::
    :label: eq:smwf_state

    \mathbf{X} =
    \begin{bmatrix}
    \mathbf{x}_1\\
    \mathbf{x}_2\\
    \end{bmatrix}=
    \begin{bmatrix}
    {}^O\mathbf{r}_{B/O} \\
    {}^O\dot{\mathbf{r}}_{B/O} \\
    \end{bmatrix}

The associated frame definitions may be found in the following table.

.. list-table:: Frame Definitions
    :widths: 25 25
    :header-rows: 1

    * - Frame Description
      - Frame Definition
    * - Small Body Hill Frame
      - :math:`O: \{\hat{\mathbf{o}}_1, \hat{\mathbf{o}}_2, \hat{\mathbf{o}}_3\}`
    * - Spacecraft Body Frame
      - :math:`B: \{\hat{\mathbf{b}}_1, \hat{\mathbf{b}}_2, \hat{\mathbf{b}}_3\}`

The derivation of the control law is skipped here for brevity. The thrust, however, is computed as follows:

.. math::
    :label: eq:smwf_u

    \begin{equation}
    \mathbf{u} = -(f(\mathbf{x}) - f(\mathbf{x}_{ref})) - [K_1]\Delta\mathbf{x}_1 - [K_2]\Delta\mathbf{x}_2
    \end{equation}

The relative velocity dynamics are described in detail by `Takahashi <https://doi.org/10.2514/1.G005733>`__ and
`Scheeres <http://dx.doi.org/10.2514/1.57247>`__.

.. math::
    :label: eq:smwf_x_dot_2

    \begin{split}
    f(\mathbf{x}) = ^O\ddot{\mathbf{r}}_{S/O} = -\ddot{F}[\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_1 - 2\dot{F}[\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_2 - \dot{F}^2[\tilde{\hat{\mathbf{o}}}_3][\tilde{\hat{\mathbf{o}}}_3]\mathbf{x}_1- \dfrac{\mu_a \mathbf{x}_1}{||\mathbf{x}_1||^3} + \dfrac{\mu_s(3{}^O\hat{\mathbf{d}}{}^O\hat{\mathbf{d}}^T-[I_{3 \times 3}])\mathbf{x}_1}{d^3} \\
    + C_{SRP}\dfrac{P_0(1+\rho)A_{sc}}{M_{sc}}\dfrac{(1\text{AU})^2}{d^2}\hat{\mathbf{o}}_1 + \sum_i^I\dfrac{{}^O\mathbf{F}_i}{M_{sc}} + \sum_j^J\dfrac{{}^O\mathbf{F}_j}{M_{sc}}
    \end{split}


User Guide
^^^^^^^^^^
A detailed example of the module is provided in :ref:`scenarioSmallBodyFeedbackControl`. However, the initialization
of the module is also shown here. The module is first initialized as follows:

.. code-block:: python

    waypointFeedback = smallBodyWaypointFeedback.SmallBodyWaypointFeedback()

The asteroid ephemeris input message is then connected. In this example, we use the :ref:`planetNav` module.

.. code-block:: python

    waypointFeedback.asteroidEphemerisInMsg.subscribeTo(planetNavMeas.ephemerisOutMsg)

A standalone message is created for the sun ephemeris message.

.. code-block:: python

    sunEphemerisMsgData = messaging.EphemerisMsgPayload()
    sunEphemerisMsg = messaging.EphemerisMsg()
    sunEphemerisMsg.write(sunEphemerisMsgData)
    waypointFeedback.sunEphemerisInMsg.subscribeTo(sunEphemerisMsg)

The navigation attitude and translation messages are then subscribed to

.. code-block:: python

    waypointFeedback.navAttInMsg.subscribeTo(simpleNavMeas.attOutMsg)
    waypointFeedback.navTransInMsg.subscribeTo(simpleNavMeas.transOutMsg)

Finally, the area, mass, inertia, and gravitational parameter of the asteroid are initialized

.. code-block:: python

    waypointFeedback.A_sc = 1.  # Surface area of the spacecraft, m^2
    waypointFeedback.M_sc = mass  # Mass of the spacecraft, kg
    waypointFeedback.IHubPntC_B = unitTestSupport.np2EigenMatrix3d(I)  # sc inertia
    waypointFeedback.mu_ast = mu  # Gravitational constant of the asteroid

The reference states are then defined:

.. code-block:: python

    waypointFeedback.x1_ref = [-2000., 0., 0.]
    waypointFeedback.x2_ref = [0.0, 0.0, 0.0]

Finally, the feedback gains are set:

.. code-block:: python

    waypointFeedback.K1 = unitTestSupport.np2EigenMatrix3d([5e-4, 0e-5, 0e-5, 0e-5, 5e-4, 0e-5, 0e-5, 0e-5, 5e-4])
    waypointFeedback.K2 = unitTestSupport.np2EigenMatrix3d([1., 0., 0., 0., 1., 0., 0., 0., 1.])