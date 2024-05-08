Executive Summary
-----------------
This module maps commanded forces and torques defined in the body frame of the spacecraft to a set of thrusters. It is
capable of handling Center of Mass (CoM) offsets and non-controllable axis.  In contrast to :ref:`thrForceMapping`, this module
only handles on-pulsing, but not off-pulsing.  Further, it provides a single force/torque projection onto the thrusters and
thus lacks some of the robustness features of :ref:`thrForceMapping`.

The commanded force and torque input messages are optional, and the associated vectors are zeroed if no
input message is connected.  This provides a general capability to map control torques, forces or torques and forces
onto a set of thrusters.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  
The module msg connection is set by the user from python.  
The msg type contains a link to the message structure definition, while the description 
provides information on what this message is used for.
Both the `cmdTorqueInMsg` and `cmdForceInMsg` are optional.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - cmdTorqueInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - (optional) The name of the vehicle control torque (Lr) input message
    * - cmdForceInMsg
      - :ref:`CmdForceBodyMsgPayload`
      - (optional) The name of the vehicle control force input message
    * - thrConfigInMsg
      - :ref:`THRArrayConfigMsgPayload`
      - The name of the thruster cluster input message
    * - vehConfigInMsg
      - :ref:`VehicleConfigMsgPayload`
      - The name of the vehicle config input message
    * - thrForceCmdOutMsg
      - :ref:`THRArrayCmdForceMsgPayload`
      - The name of the output thruster force message

Detailed Module Description
---------------------------
Force and Torque Mapping
^^^^^^^^^^^^^^^^^^^^^^^^
The desired force and torque are given as :math:`\mathbf{F}_{req}` and :math:`\boldsymbol{\tau}_{req}`, respectively.
These are both stacked into a single vector.

.. math::
    :label: eq:augmented_ft

    \begin{bmatrix}
    \boldsymbol{\tau}_{req} \\
    \mathbf{F}_{req}
    \end{bmatrix}

The :math:`i_{th}` thruster position expressed in spacecraft body-fixed coordinates is given by :math:`\mathbf{r}_i`. The
unit direction vector of the thruster force is :math:`\hat{\mathbf{g}}_{t_i}`. The thruster force is given as:

.. math::
    :label: eq:force_direction

    \mathbf{F}_i = F_i\hat{\mathbf{g}}_{t_i}

The torque produced by each thruster about the body-fixed CoM is:

.. math::
    :label: eq:torques

    \boldsymbol{\tau}_i = ((\mathbf{r}_i - \mathbf{r}_{\text{COM}}) \times \hat{\mathbf{g}}_{t_i})F_i = \mathbf{d}_iF_i

The total force and torque on the spacecraft may be represented as:

.. math::
    :label: eq:sys_eqs

    \begin{bmatrix}
        \boldsymbol{\tau}_{req} \\
        \mathbf{F}_{req}
    \end{bmatrix} =
    \begin{bmatrix}
        \mathbf{d}_i \ldots \mathbf{d}_N \\
        \hat{\mathbf{g}}_{t_i} \ldots \hat{\mathbf{g}}_{t_N}
    \end{bmatrix}
    \begin{bmatrix}
        F_1 \\
        \vdots \\
        F_N
    \end{bmatrix} = [D]\mathbf{F}

The force required by each thruster can computed by the following equation. Any rows within the :math:`[D]` matrix
that contain only zeros are removed beforehand.

.. math::
    :label: eq:soln

    \mathbf{F} = [D]^T([D][D]^T)^{-1}\begin{bmatrix}
                                         \boldsymbol{\tau}_{req} \\
                                         \mathbf{F}_{req}
                                         \end{bmatrix}

To ensure no commanded thrust is less than zero, the minimum thrust is subtracted from the thrust vector

.. math::
    :label: eq:F_min

    \mathbf{F} = \mathbf{F} - \text{min}(\mathbf{F})

These thrust commands are then written to the output message.


User's Guide
------------
To set up this module users must create the config data and module wrap::

    module = forceTorqueThrForceMapping.forceTorqueThrForceMapping()
    module.ModelTag = "forceTorqueThrForceMappingTag"
    unitTestSim.AddModelToTask(unitTaskName, module)

The ``cmdForceInMsg`` and ``cmdTorqueInMsg`` are optional. However, the ``thrConfigInMsg`` and ``vehConfigInMsg`` are not. These
can both be set up as follows, where ``rcsLocationData`` is a list of the thruster positions and ``rcsDirectionData`` is a
list of thruster directions. ``CoM_B`` is the center of mass of the spacecraft in the body frame.::

    fswSetupThrusters.clearSetup()
    for i in range(numThrusters):
        fswSetupThrusters.create(rcsLocationData[i], rcsDirectionData[i], maxThrust)
    thrConfigInMsg = fswSetupThrusters.writeConfigMessage()
    vehConfigInMsgData = messaging.VehicleConfigMsgPayload()
    vehConfigInMsgData.CoM_B = CoM_B
    vehConfigInMsg = messaging.VehicleConfigMsg().write(vehConfigInMsgData)

Then, the relevant messages must be subscribed to by the module::

    module.cmdTorqueInMsg.subscribeTo(cmdTorqueInMsg)
    module.cmdForceInMsg.subscribeTo(cmdForceInMsg)
    module.thrConfigInMsg.subscribeTo(thrConfigInMsg)
    module.vehConfigInMsg.subscribeTo(vehConfigInMsg)

For more information on how to set up and use this module, see the unit test.
