Executive Summary
-----------------
This module converts two inertial spacecraft positions into a Hill-frame relative position and velocity.

Message Connection Descriptions
-------------------------------


.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - chiefStateInMsg
      - :ref:`NavTransMsgPayload`
      - Chief inertial state navigation message used as basis for Hill frame definition
    * - depStateInMsg
      - :ref:`NavTransMsgPayload`
      - Deputy inertial state navigation message.
    * - hillStateOutMsg
      - :ref:`HillRelStateMsgPayload`
      - Relative position message of deputy w.r.t. chief in Hill-frame coordinates.

Detailed Module Description
---------------------------

This module converts a pair of spacecraft inertial positions into a relative position and velocity represented in the chief spacecraft's hill frame.

The Hill frame relative position and velocity of the deputy spacecraft is returned in a :ref:`HillRelStateMsgPayload`. 

The chief Hill frame is computed from the Chief navigation message by first calculating the radial and orbit angular momentum unit vectors:

    .. math:: 
        \hat{h}_r = \frac{r}{|r|}

    .. math:: 
        \hat{h}_h = \frac{r \times v}{|r \times v|}

These unit vectors are used to compute the final direction used to define the frame via the cross product:

    .. math::
        \hat{h}_v = \hat{h}_h \times \hat{h}_r

Finally, these unit vectors are composed into a DCM that rotates from the inertial to Hill frames:

    .. math:: 
        [HN] = \{\hat{h}_r, \hat{h}_v, \hat{h}_h\}

The relative position is computed using the difference of the inertial deputy and chief positions and this DCM:

.. math::
    \rho = [HN](r_{dep} - r_{chief})

To compute the relative velocities, the transport theorem is additionally used due to the rotating nature of the Hill frame:

    .. math::
        \dot{\rho} = [HN](\dot{r}_{dep} - \dot{r}_{chief}) + (\omega_{HN} \times (\dot{r}_{dep} - \dot{r}_{chief}))

Module Assumptions and Limitations
----------------------------------
This module makes use of the rv2hill function provided by the orbitalMotion library included with BSK, which works for arbitrary
orbit geometries (i.e., it does not assume circular orbits in calculating the velocity transformation). As a result, it is broadly applicable
to computing relative orbit states.


User Guide
----------
This module is configured as a message transformer and has no user-settable parameters aside from the message inputs and outputs.

A simple example of this module's initialization alongside a recorder to store the relative state information is provided here::

    .. code-block:: python
        :linenos:
        hillStateNavData = hillStateConverter.hillStateConverter()
        hillStateNavData.ModelTag = "dep_hillStateNav"
        hillStateNavData.chiefStateInMsg.subscribeTo(chiefNavMsg)
        hillStateNavData.depStateInMsg.subscribeTo(depNavMsg)
        hillRecorder = hillStateNavData.hillStateOutMsg.recorder()

In addition, this module is used in the example script :ref:`scenarioDragRendezvous` as an input to the :ref:`hillToAttRef` module.
