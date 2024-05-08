Executive Summary
-----------------

This module computes a Body frame reequested dipole given a requested body torque and magnetic field vector.

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
    * - tamSensorBodyInMsg
      - :ref:`TAMSensorBodyMsgPayload`
      - input message for magnetic field sensor data
    * - tauRequestInMsg
      - :ref:`CmdTorqueBodyMsgPayload`
      - input message containing control torque in the Body frame
    * - dipoleRequestOutMsg
      - :ref:`DipoleRequestBodyMsgPayload`
      - output message containing dipole request in the Body frame

Detailed Module Description
---------------------------

The following presents the mathematics for converting a requested Body torque into a requested Body dipole to be produced the torque rods given the local magnetic field vector. The desired Body frame dipole is given by

.. math::
    {}^{\cal B} {\pmb\mu}_{\text{desired}} = \frac{1}{|\bf b|^2}
    {}^{\cal B}{\bf b} \times \ {}^{\cal B} {\pmb\tau}_{\text{desired}} = [G_t] {\pmb\mu}_{\text{cmd}}

where :math:`\bf b` is the local magnetic field vector and :math:`[G_t]` is a 3 :math:`\times N_{\text{MTB}}`
matrix that transforms the individual rod dipoles to the Body frame.

User Guide
----------
See the example script :ref:`scenarioMtbMomentumManagementSimple` for an illustration on how to use this module.
