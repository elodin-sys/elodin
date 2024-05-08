Executive Summary
-----------------

Module to compute the Inertial-3D spinning pointing navigation solution.  This spin can be relative to an inertial
frame or relative to an input reference frame message.


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
      - attitude reference output message
    * - attRefInMsg
      - :ref:`AttRefMsgPayload`
      - (optional) attitude reference input message, if not connected then a zeroed input reference state is set

Module Description
------------------
This module generates an attitude reference frame state output message where the reference frame is rotating at
a constant angular velocity vector relative to the input frame :math:`R_0`.  If the input attitude frame message
is not connected, then a zero'd input message is assumed.

The input reference frame :math:`R_0` (user defined input message or default zero'd message)
contains the state information in the form :math:`\pmb\sigma_{R_0/N}`, :math:`\pmb\omega_{R_0/N}` and
:math:`\dot{\pmb\omega}_{R_0/N}`.  The constant spin vector is constant with respect to :math:`R_0`
and given by :math:`{}^{R_0}{\pmb\omega}_{R/R0}`.

The angular velocity of the of the output reference frame :math:`R` is then given by

.. math::  \pmb \omega_{R/N} = \pmb\omega_{R/R_0} + \pmb\omega_{R_0/N}

As the spin vector :math:`{}^{R_0}{\pmb\omega}_{R/R0}` is constant with respect to :math:`R_0`, then
the output frame angular acceleration is

.. math:: \dot{\pmb\omega}_{R/N} = {\pmb\omega}_{R_0/N} \times {\pmb\omega}_{R/R0} + \dot{\pmb\omega}_{R_0/N}

Finally, the output frame MRP orientation is computed using

.. math:: \dot{\pmb\sigma}_{R/N} = \frac{1}{4} [B(\pmb\sigma_{R/N})]\ {}^{R}{\pmb\omega}_{R/N}

where

.. math:: [B(\pmb\sigma_{R/N})] = (1-\sigma_{R/N}^{2}) [I_{3\times 3}] + 2 [\tilde{\pmb\sigma}_{R/N}] + 2 \pmb\sigma_{R/N} \pmb\sigma_{R/N}^{T}

with :math:`\sigma_{R/N} = |\pmb\sigma_{R/N}|`.


User Guide
----------
The only parameter that must be set is ``omega_RR0_R0`` representing the :math:`R_0`-constant spin axis
:math:`{}^{R_0}{\pmb\omega}_{R/R0}`.


