Executive Summary
-----------------

This module is used to calculate the required rotation angle for a solar array that is able to rotate about its drive axis. The degree of freedom associated with the rotation of the array about the drive axis makes it such that it is possible to improve the incidence angle between the sun and the array surface, thus ensuring maximum power generation.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the user from python.  The msg type contains a link to the message structure definition, while the description provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - hingedRigidBodyRefOutMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Output Hinged Rigid Body Reference Message.
    * - attNavInMsg
      - :ref:`NavAttMsgPayload`
      - Input Attitude Navigation Message.
    * - attRefInMsg
      - :ref:`AttRefMsgPayload`
      - Input Attitude Reference Message.
    * - hingedRigidBodyInMsg
      - :ref:`HingedRigidBodyMsgPayload`
      - Input Hinged Rigid Body Message Message.


Module Assumptions and Limitations
----------------------------------
This module computes the rotation angle required to achieve the best incidence angle between the Sun direction and the solar array surface. This does not mean that
perfect incidence (Sun direction perpendicular to array surface) is guaranteed. This module assumes that the solar array has only one surface that is able to generate power. This bounds the output reference angle :math:`\theta_R` between :math:`0` and :math:`2\pi`. Perfect incidence is achievable when the solar array drive direction and the Sun direction are perpendicular. Conversely, when they are parallel, no power generation is possible, and the reference angle is set to the current angle, to avoid pointless energy consumption attempting to rotate the array.

The Sun direction in body-frame components is extracted from the ``attNavInMsg``. The output reference angle :math:`\theta_R`, however, can be computed either based on the reference attitude contained in ``attRefInMsg``, or the current spacecraft attitude contained also in ``attNavInMsg``. This depends on the frequency with which the arrays need to be actuated, in comparison with the frequency with which the motion of the spacecraft hub is controlled. The module input ``attitudeFrame`` allows the user to set whether to compute the reference angle based on the reference attitude or current spacecraft attitude.


Detailed Module Description
---------------------------
For this module to operate, the user needs to provide two unit directions as inputs:

- :math:`{}^\mathcal{B}\boldsymbol{\hat{a}}_1`: direction of the solar array drive, about which the rotation happens;
- :math:`{}^\mathcal{B}\boldsymbol{\hat{a}}_2`: direction perpendicular to the solar array surface, with the array at a zero rotation.

To compute the reference rotation :math:`\theta_R`, the module computes the unit vector :math:`{}^\mathcal{R}\boldsymbol{\hat{a}}_2`, which is coplanar with 
:math:`{}^\mathcal{B}\boldsymbol{\hat{a}}_1` and the Sun direction :math:`{}^\mathcal{R}\boldsymbol{\hat{r}}_S`. This is obtained as:

.. math::
    {}^\mathcal{R}\boldsymbol{a}_2 = {}^\mathcal{R}\boldsymbol{\hat{r}}_S - ({}^\mathcal{R}\boldsymbol{\hat{r}}_S \cdot {}^\mathcal{B}\boldsymbol{\hat{a}}_1) {}^\mathcal{B}\boldsymbol{\hat{a}}_1

and then normalizing to obtain :math:`{}^\mathcal{R}\boldsymbol{\hat{a}}_2`. The reference angle :math:`\theta_R` is the angle between :math:`{}^\mathcal{B}\boldsymbol{\hat{a}}_2` and :math:`{}^\mathcal{R}\boldsymbol{\hat{a}}_2`:

.. math::
    \theta_R = \arccos ({}^\mathcal{B}\boldsymbol{\hat{a}}_2 \cdot {}^\mathcal{R}\boldsymbol{\hat{a}}_2).

The same math applies to the case where the body reference is used. In that case, the same vectors are expressed in body-frame coordinates. Note that the unit directions :math:`\boldsymbol{\hat{a}}_i` have the same components in both the body and reference frame, because they are body-fixed and rotate with the spacecraft hub.

Some logic is implemented such that the computed reference angle :math:`\theta_R` and the current rotation angle :math:`\theta_C` received as input from the ``hingedRigidBodyInMsg`` are never more than 360 degrees apart.

The derivative of the reference angle :math:`\dot{\theta}_R` is computed via finite differences.


User Guide
----------
The required module configuration is::

    solarArray = solarArrayRotation.solarArrayRotation()
    solarArray.ModelTag = "solarArrayRotation"  
    solarArray.a1Hat_B = [1, 0, 0]
    solarArray.a2Hat_B = [0, 0, 1]
    solarArray.attitudeFrame = 0
    unitTestSim.AddModelToTask(unitTaskName, solarArray)
	
The module is configurable with the following parameters:

.. list-table:: Module Parameters
   :widths: 34 66
   :header-rows: 1

   * - Parameter
     - Description
   * - ``a1Hat_B``
     - solar array drive direction in B-frame coordinates
   * - ``a2Hat_B``
     - solar array zero-rotation direction, in B-frame coordinates
   * - ``attitudeFrame``
     - 0 for reference angle computed w.r.t reference frame; 1 for reference angle computed w.r.t. body frame; defaults to 0 if not specified
