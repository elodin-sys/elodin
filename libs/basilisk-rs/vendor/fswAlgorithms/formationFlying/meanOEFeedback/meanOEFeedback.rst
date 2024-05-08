Summary
-------
The primary purpose of this module is to calculate feedback control input for deputy spacecraft in terms of mean orbital element difference.
This module uses Lyapunov control theory described in chapter 14 of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__.
In addition to classic orbital element set :math:`(a,e,i,\Omega,\omega,M)`, this module can also deal with equinoctial orbital element
:math:`(a,e\sin{(\omega+\Omega)},e\cos{(\omega+\Omega)},\tan{(i/2)}\sin{(\Omega)},\tan{(i/2)}\cos{(\Omega)},\Omega+\omega+M)` in order to avoid singularity.
A control input :math:`\bf{u}` used in this module is described in Eq. :eq:`eq:control_input`.

.. math::
    \bf{u} = -[B(\bf{oe_d})]^T[K]\Delta \bf{oe}
    :label: eq:control_input

Control matrix :math:`B(oe)` derived from Gauss's Planetary Equation is given in Eq.(14.213) of `Analytical Mechanics of Space Systems <http://doi.org/10.2514/4.105210>`__ for classic orbital element,
or in `Application of several control techniques for the ionospheric observation nanosatellite formation <https://www.researchgate.net/publication/228703564_Application_of_several_control_techniques_for_the_ionospheric_observation_nanosatellite_formation>`__
for equinoctial one. Depending on preference, you can change which orbital element set to use by setting :math:`oe_{type}` parameter.
In order to calculate mean oe, position and velocity input messages are converted to
osculating oe. Then, osculating oe is converted to mean oe by Brower's theory.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_meanOEFeedback:

.. table:: Module I/O Messages
    :widths: 25 25 100

    +-----------------------+-----------------------------------+---------------------------------------------------------------+
    | Msg Variable Name     | Msg Type                          | Description                                                   |
    +=======================+===================================+===============================================================+
    | chiefTransInMsg       | :ref:`NavTransMsgPayload`         | The name of the chief's position and velocity input message   |
    +-----------------------+-----------------------------------+---------------------------------------------------------------+
    | deputyTransInMsg      | :ref:`NavTransMsgPayload`         | The name of the deputy's position and velocity input message  |
    +-----------------------+-----------------------------------+---------------------------------------------------------------+
    | forceOutMsg           | :ref:`CmdForceInertialMsgPayload` | Calculated Force to control orbital element difference        |
    |                       |                                   | output message                                                |
    +-----------------------+-----------------------------------+---------------------------------------------------------------+

Module Assumptions and Limitations
----------------------------------
- This module assumes that target orbital element differnce is constant during simulation.

User Guide
----------------------------------
This module requires the following variables to be set as parameters:

- ``oeType`` 0 for classic oe (default), 1 for equinoctial oe
- ``mu`` gravitational constant for a central body in m^3/s^2
- ``req`` equatorial radius of a central body in meters
- ``J2`` J2 constant of a central body
- ``targetDiffOeMean`` desired mean orbital element difference. Assumed oe is linked to ``oeType``.
- ``K`` control gain matrix

You must be careful about a meter unit used for ``mu`` and ``req``.
For ``targetDiffOeMean``, normalized semi major axis must be used.
