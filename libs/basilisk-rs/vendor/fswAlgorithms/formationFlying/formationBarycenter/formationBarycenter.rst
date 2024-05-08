Executive Summary
-----------------
This module computes the barycenter of a swarm of satellites. The barycenter can either be computed in the regular cartesian way (using a weighted average of the position and velocity vectors) 
or using the weighted average of the orbital elements. Both output two navigation messages that describe the position and velocity of the barycenter. The output messages contain the same 
information, although one is a C++ message and the other is a C-wrapped message.

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
    * - scNavInMsgs
      - :ref:`NavTransMsgPayload`
      - vector of spacecraft navigation input messages.  These are set through ``addSpacecraftToModel()``
    * - scPayloadInMsgs
      - :ref:`VehicleConfigMsgPayload`
      - vector of spacecraft payload input messages.  These are set through ``addSpacecraftToModel()``
    * - transOutMsg
      - :ref:`NavTransMsgPayload`
      - barycenter information C++ output message
    * - transOutMsgC
      - :ref:`NavTransMsgPayload`
      - barycenter information C-wrapped output message

Detailed Module Description
---------------------------

This module computes the barycenter of a swarm of spacecraft. For the cartesian method, a simple center of mass calculation is made for the position and velocity vectors. 
Let :math:`\textbf{x}` represent either the position or the velocity vectors. The corresponding weighted average is:

.. math::
    \bar{\textbf{x}} = \dfrac{1}{m_{total}}\sum_{i}m_i\textbf{x}_i,

where :math:`m_{total}=\sum_{i}m_i`.

For the orbital elements averaging, the process is similar. However, the position and velocity vectors of each spacecraft must first be converted to orbital elements. Once 
that is done, we take the average of each orbital element :math:`oe` as such:

.. math::
    \bar{oe} = \dfrac{1}{m_{total}}\sum_{i}m_ioe_i

This formula is only valid for semi-major axis (a), eccentricity (e) and inclination (i). For the other angular orbital elements, a problem with angle wrapping can occur 
when the angles are close to zero. For example, if two spacecraft of equal mass have a true anomaly of 10 and 350 degrees, the previous averaging formula would suggest 
that the mean should be 180 degrees, when in fact it should be 0. To solve this problem, a different formula is used for RAAN (:math:`\Omega`), AoP (:math:`\omega`) 
and true anomaly (f):

.. math::
    \bar{\alpha} = \texttt{atan2}\left(\sum_{i}m_i\sin\alpha_i, \sum_{i}m_i\cos\alpha_i\right)

where :math:`\bar{\alpha}` is the averaged angular orbital element. The set of :math:`\bar{oe}` are then converted back into position and velocity vectors.

As stated before, the module outputs both a C++ and C-wrapped navigation messages. Both contain the same payload.

Model Assumptions and Limitations
---------------------------------

This code makes the following assumptions:

- **Gravitational parameter is known** 

This code has the following limitations:

- **Equatorial singularities**: when using orbital element averaging, near equatorial orbits may have induce a singularity in the ascending node.


User Guide
----------

This section contains conceptual overviews of the code and clear examples for the prospective user.

Module Setup
~~~~~~~~~~~~

The temperature module is created in python using:

.. code-block:: python
    :linenos:

    barycenterModule = formationBarycenter.FormationBarycenter()
    barycenterModule.ModelTag = 'barycenter'

A sample setup is done using:

.. code-block:: python
    :linenos:

    # Configure spacecraft state input messages
    scNavMsgData1 = messaging.NavTransMsgPayload()
    scNavMsgData1.r_BN_N = rN1
    scNavMsgData1.v_BN_N = vN1
    scNavMsg1 = messaging.NavTransMsg().write(scNavMsgData1)

    scNavMsgData2 = messaging.NavTransMsgPayload()
    scNavMsgData2.r_BN_N = rN2
    scNavMsgData2.v_BN_N = vN2
    scNavMsg2 = messaging.NavTransMsg().write(scNavMsgData2)

    # Configure spacecraft mass input messages
    scPayloadMsgData1 = messaging.VehicleConfigMsgPayload()
    scPayloadMsgData1.massSC = 100
    scPayloadMsg1 = messaging.VehicleConfigMsg().write(scPayloadMsgData1)

    scPayloadMsgData2 = messaging.VehicleConfigMsgPayload()
    scPayloadMsgData2.massSC = 150
    scPayloadMsg2 = messaging.VehicleConfigMsg().write(scPayloadMsgData2)

    # add spacecraft input messages to module
    barycenterModule.addSpacecraftToModel(scNavMsg1, scPayloadMsg1)
    barycenterModule.addSpacecraftToModel(scNavMsg2, scPayloadMsg2)

No further setup is needed for the cartesian method. If the user wants to use orbital elements, the following additional code is needed:

.. code-block:: python
    :linenos:

    barycenterModule.useOrbitalElements = True
    barycenterModule.mu = mu

