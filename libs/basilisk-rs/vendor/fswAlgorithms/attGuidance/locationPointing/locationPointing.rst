Executive Summary
-----------------
This module generates an attitude guidance message to make a specified spacecraft pointing vector target an inertial location.
This location could be on a planet if this module is connected with :ref:`groundLocation` for example, or it could point
to a celestial object center using :ref:`EphemerisMsgPayload`, or a spacecraft location using :ref:`NavTransMsgPayload`.

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
    * - scAttInMsg
      - :ref:`NavAttMsgPayload`
      - input msg with inertial spacecraft attitude states
    * - scTransInMsg
      - :ref:`NavTransMsgPayload`
      - input msg with inertial spacecraft translational states
    * - locationInMsg
      - :ref:`GroundStateMsgPayload`
      - input msg containing the inertial point location of interest
    * - celBodyInMsg
      - :ref:`EphemerisMsgPayload`
      - (alternative) input msg containing the inertial point location of a celestial body of interest
    * - scTargetInMsg
      - :ref:`NavTransMsgPayload`
      - (alternative) input msg with inertial target spacecraft translational states
    * - attGuidOutMsg
      - :ref:`AttGuidMsgPayload`
      - output message with the attitude guidance
    * - attRefOutMsg
      - :ref:`AttRefMsgPayload`
      - output message with the attitude reference



Detailed Module Description
---------------------------
The inertial location of interest is given by :math:`{\bf r}_{L/N}` and can be either extracted from ``locationInMsg`` when 
a location on a planet is provided,  ``celBodyInMsg`` when a celestial body's ephemeris location is provided (for pointing
at the Sun or the Earth), or ``scTargetInMsg`` when pointing at another spacecraft.
The vector pointing from the satellite location :math:`{\bf r}_{S/N}` to this location is then

.. math::
    {\bf r}_{L/S} = {\bf r}_{L/N} - {\bf r}_{S/N}

Let :math:`\hat{\bf r}_{L/S}` be the normalized heading vector to this location.

The unit vector :math:`\hat{\bf p}` is a body-fixed vector and denotes the body axis which is to point towards
the desired location :math:`L`.  Thus this modules performs a 2-degree of freedom attitude guidance and
control solution.

The eigen-axis to rotate :math:`\hat{\bf p}` towards :math:`\hat{\bf r}_{L/S}` is given by

.. math::

    \hat{\bf e} = \frac{\hat{\bf p} \times \hat{\bf r}_{L/S}}{|\hat{\bf p} \times \hat{\bf r}_{L/S}|}

The principle rotation angle :math:`\phi` is

.. math::

    \phi = \arccos (\hat{\bf p} \cdot \hat{\bf r}_{L/S} )

The attitude tracking error :math:`{\pmb\sigma}_{B/R}` is then given by

.. math::

    {\pmb\sigma}_{B/R} = - \tan(\phi/4) \hat{\bf e}

The tracking error rates :math:`{\pmb\omega}_{B/R}` are obtained through numerical differentiation of the
MRP values.  During the first module ``Update`` evaluation the numerical differencing is not possible and
this value is thus set to zero.

Using the attitude navigation and guidance messages, this module also computes the reference information in 
the form of ``attRefOutMsg``. This additional output message is useful when working with modules that need 
a reference message and cannot accept a guidance message.

.. note::

    The module checks for several conditions such as heading vectors
    being collinear, the MRP switching during the numerical differentiation, etc.



User Guide
----------
The one required variable that must be set is ``pHat_B``.  This is body-fixed unit vector which is to be
pointed at the desired inertial location.

The user should only connect one location of interest input message, either ``locationInMsg``, ``celBodyInMsg`` or ``scTargetInMsg``. 
Connecting both will result in a warning and the module defaults to using the ``locationInMsg`` information.

This 2D attitude control module provides two output messages in the form of :ref:`attGuidMsgPayload` and :ref:`attRefMsgPayload`.
The first guidance message, describing body relative to reference tracking errors,
can be directly connected to an attitude control module.  However, at times we need to have the
attitude reference message as the output to feed to :ref:`attTrackingError`.  Here the ``B/R`` states are subtracted
from the ``B/N`` states to obtain the equivalent ``R/N`` states.

The variable ``smallAngle`` defined the minimum angular separation where two vectors are considered colinear.
It is defaulted to zero, but can be set to any desired value in radians.

By default this is a 2D attitude control module in attitude and a 2D rate control.  In particular, the rates about the
desired heading axis are not damped.  By setting the module variable ``useBoresightRateDamping`` to 1,
the body rates about about the desired heading 
angle are added to the rate tracking error yielding a 3D rate control implementation.  
