Executive Summary
-----------------

This module computes individual torque rod dipole commands taking into account saturation limits.

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
    * - mtbParamsInMsg
      - :ref:`MTBArrayConfigMsgPayload`
      - input message for MTB layout
    * - dipoleRequestBodyInMsg
      - :ref:`DipoleRequestBodyMsgPayload`
      - input message containing the requested body frame dipole
    * - dipoleRequestMtbOutMsg
      - :ref:`MTBCmdMsgPayload`
      - input message containing the individual dipole requests for each torque bar on the vehicle

Detailed Module Description
---------------------------
The following presents the mathematics for mapping a Body frame dipole request into individual torque rod dipole commands.

The individual rod dipoles are given by

.. math::
    {\pmb \mu}_{\text{cmd}} = [G_t]^{\dagger} \ {}^{\cal B} {\pmb\mu}_{\text{desired}}

where the :math:`\dagger` symbol denotes the pseudo inverse. The dipole commands may need to be
saturated at this point. The saturated commands are referred to as :math:`{\pmb\mu}_{\text{saturated}}`
from here on out in this document.

User Guide
----------
See the example script :ref:`scenarioMtbMomentumManagementSimple` for an illustration on how to use this module.

Note that user must set the torque rod alignment matrix ``GtMatrix_B`` and the ``steeringMatrix`` in row major format.
Also note that ``steeringMatrix`` is simply the psuedoinverse of the torque rod alignment matrix
