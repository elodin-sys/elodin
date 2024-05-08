Executive Summary
-----------------
This module computes the control thrust force of the Electrostatic Tractor Relative Motion Control. A servicing
satellite and a debris (or other satellite) are charged to different electrostatic potentials, resulting in an
attractive force between the two craft. The Electrostatic Tractor (ET) concept uses this attractive force to tug the debris to another orbit. See `Relative Motion Control For Two-Spacecraft Electrostatic Orbit Corrections <https://doi.org/10.2514/1.56118>`__ for more information on the ET relative motion control.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_ET_spherical_control:
.. figure:: /../../src/fswAlgorithms/formationFlying/etSphericalControl/_Documentation/Images/moduleEtSphericalControl.svg
    :align: center

    Figure 1: ``etSphericalControl()`` Module I/O Illustration


.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - servicerTransInMsg
      - :ref:`NavTransMsgPayload`
      - Servicer position and velocity input message
    * - debrisTransInMsg
      - :ref:`NavTransMsgPayload`
      - Debris position and velocity input message
    * - servicerAttInMsg
      - :ref:`NavAttMsgPayload`
      - Servicer attitude input message
    * - servicerVehicleConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - Servicer vehicle configuration (mass information) input message
    * - debrisVehicleConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - Debris vehicle configuration (mass information) input message
    * - eForceInMsg
      - :ref:`CmdForceInertialMsgPayload`
      - Inertial electrostatic force that acts on servicer input message
    * - forceInertialOutMsg
      - :ref:`CmdForceInertialMsgPayload`
      - Inertial frame control thrust force output message
    * - forceBodyOutMsg
      - :ref:`CmdForceBodyMsgPayload`
      - Body frame control thrust force output message

Module Assumptions and Limitations
----------------------------------
This control law in this module is based on an attractive electrostatic force between the two craft, so the electric potentials of the two craft must be different in sign (assuming that both craft are fully conducting).

Equations
^^^^^^^^^
The necessary equations for this module are given in `Relative Motion Control For Two-Spacecraft Electrostatic Orbit
Corrections <https://doi.org/10.2514/1.56118>`__. Note that Eq. (45) in this paper should be

.. math:: {\bf T}_t = m_T [ {\bf u} - {\bf F}_c (1/m_T + 1/m_D)]

User Guide
----------
The ETcontrol module is created using:

.. code-block:: python
    :linenos:

    etSphericalControl = etSphericalControl.etSphericalControl()
    etSphericalControl.ModelTag = "etSphericalControl"
    scSim.AddModelToTask(fswTaskName, etSphericalControl, etSphericalControl)

The reference position variables in the spherical frame :math:`L_r`, :math:`theta_r`, :math:`phi_r`,
the feedback gains :math:`K` and :math:`P`, and the gravitational parameter mu must
be added to etSphericalControl.

The module computes the control force vector both with respect to the inertial and body frame as
separate output messages.  Depending on the needs of the developer, the control force can be connected
in either frame to down-stream modules.  However, don't connect both output messages because
this would result in the control force being applied twice.