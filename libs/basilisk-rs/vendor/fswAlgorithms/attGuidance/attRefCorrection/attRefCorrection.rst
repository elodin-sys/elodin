Executive Summary
-----------------
This module reads in the attitude reference message and adjusts it by a fixed rotation.  This allows a general body-fixed frame
:math:`B` to align with this corrected reference frame :math:`R_c`.

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
    * - attRefInMsg
      - :ref:`AttRefMsgPayload`
      - attitude reference input message
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - corrected attitude reference input message

Detailed Module Description
---------------------------

This module is an attitude reference message feed-through module where a fixed orientation offset can be applied
to the output attitude ``sigma_RN``.  In not all cases do we wish to drive a body-fixed
frame :math:`\cal B` to a reference frame :math:`\cal R`.  Rather, maybe it is desired to align
the first :math:`\cal R` frame axis with the 2nd body axis.  Thus, a corrected body frame :math:`{\cal B}_c`
must align with R.  The can also be achieved by aligning :math:`\cal B` with a corrected attitude
reference frame :math:`{\cal R}_c`.

Let the rotation between :math:`\cal B` and :math:`{\cal B}_c` be given by the MRP set :math:`\sigma_{B/B_c}`.
Using DCMs, thus we need

.. math::
    [BN] = [R_cN]

The original reference frame relates to the body frame through

.. math::

    [RN] = [B_cN][BN]

which leads to

.. math::

    [BN] = [B_cN]^T [RN]

Substituting this into the first equatino leads to the desired corrected reference frame:

.. math::

    [R_cN] = [B_cN]^T [RN]

The orientation of :math:`[R_cN]` is then translated to a MRP set for the output message.


User Guide
----------

The only variable that is set with this module is the ``sigma_BcB`` MRP to rotate from the original
body frame and the corrected frame.

