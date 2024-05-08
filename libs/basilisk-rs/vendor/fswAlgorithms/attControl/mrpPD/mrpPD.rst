Executive Summary
-----------------
This module provides a MRP based PD attitude control module. It is similar to :ref:`mrpFeedback`, but without the RW or the integral feedback option. The feedback control is able to asymptotically track a reference attitude if there are no unknown dynamics and the attitude control torque is implemented with a thruster set.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_MRPPD:
.. figure:: /../../src/fswAlgorithms/attControl/mrpPD/_Documentation/Images/moduleIOMrpPd.svg
    :align: center

    Figure 1: ``mrpPD()`` Module I/O Illustration

.. table:: Module I/O Messages
    :widths: 25 25 100

    +-------------------------------+-------------------------------+-----------------------------------------------+
    | Msg Variable Name             | Msg Type                      | Description                                   |
    +===============================+===============================+===============================================+
    | cmdTorqueOutMsg               | :ref:`CmdTorqueBodyMsgPayload`| Commanded external torque output message      |
    +-------------------------------+-------------------------------+-----------------------------------------------+
    | vehConfigInMsg                | :ref:`VehicleConfigMsgPayload`| Vehicle configuration input message           |
    +-------------------------------+-------------------------------+-----------------------------------------------+
    | guidInMsg                     | :ref:`VehicleConfigMsgPayload`| Vehicle configuration input message           |
    +-------------------------------+-------------------------------+-----------------------------------------------+

Detailed Module Description
---------------------------
This attitude feedback module using the MRP feedback control related to the control in section 8.4.1 in `Analytical Mechanics of Space Systems <http://dx.doi.org/10.2514/4.105210>`_:

.. math::
    :label: eq-mrppd-1

    {\bf L}_{r} =  -K \pmb\sigma - [P] \delta\pmb\omega   + [I](\dot{\pmb\omega}_{r} - [\tilde{\pmb\omega}]\pmb\omega_{r})
			+[\tilde{\pmb \omega}_{r}] ]
			[I]\pmb\omega   - \bf L

Note that this control solution creates an external control torque which must be produced with a cluster of thrusters.  No reaction wheel information is used here.  Further, the feedback control component is a simple proportional and derivative feedback formulation.  As shown in `Analytical Mechanics of Space Systems <http://dx.doi.org/10.2514/4.105210>`_, this control can asymptotically track a general reference trajectory given by the reference frame :math:`\cal R`.


Module Assumptions and Limitations
----------------------------------
This control assumes the spacecraft is rigid and that the inertia tensor does not vary with time.


User Guide
----------
The following parameters must be set for the module:

- ``K``: the MRP proportional feedback gain
- ``P``: the :math:`\pmb\omega` tracking error proportional feedback gain
- ``knownTorquePntB_B``: (Optional) the known external torque vector :math:`{}^{B}{\bf L}`.  The default value is a zero vector.

