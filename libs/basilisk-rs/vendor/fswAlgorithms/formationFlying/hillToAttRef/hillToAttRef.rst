Executive Summary
-----------------
This module acts as an attitude reference generator based on attitude-driven formation control laws whose gain
matrices are derived elsewhere. Specifically, it:

1.  Obtains a relative state in the form of a :ref:`HillRelStateMsgPayload`
2.  Applies a defined gain matrix to the state to obtain a relative attitude
3.  Maps that attitude given a reference attitude to a full attitude reference message
4.  Writes out a :ref:`AttRefMsgPayload` describing the current commanded attitude


Message Connection Descriptions
-------------------------------


.. table:: Module I/O Messages
        :widths: 25 25 100

        +-----------------------+---------------------------------+---------------------------------------------------+
        | Msg Variable Name     | Msg Type                        | Description                                       |
        +=======================+=================================+===================================================+
        | hillStateInMsg        | :ref:`HillRelStateMsgPayload`   | Provides state relative to chief                  |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | attRefInMsg           | :ref:`AttRefMsgPayload`         | (Optional) Provides basis for relative attitude   |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | attNavInMsg           | :ref:`NavAttMsgPayload`         | (Optional) Provides basis for relative attitude   |
        +-----------------------+---------------------------------+---------------------------------------------------+
        | attRefOutMsg          | :ref:`AttRefMsgPayload`         | Provides the attitude reference output message.   |
        +-----------------------+---------------------------------+---------------------------------------------------+


Detailed Module Description
---------------------------
This module maps from a Hill-frame relative state into an attitude that, under the presence of drag, should result in rendezvous (i.e., a minimization of the relative state.)

More details on this process can be found in this paper, `Linear Coupled Attitude-Orbit Control Through Aerodynamic Drag <https://arc.aiaa.org/doi/10.2514/1.G004521>`__.

Module Assumptions and Limitations
----------------------------------
This module assumes that the user has supplied a gain matrix that correctly maps from relative positions and velocities to 
relative attitudes such that attitude-coupled orbital dynamics will result in desired behavior. As a result, this module is best used 
with other modules that implement attitude-coupled orbital dynamics, such as :ref:`facetDragDynamicEffector`. 


User Guide
----------
This module is configured to multiply a user-provided gain matrix by an evolving relative Hill-frame state. As such, 
this module requires the user to provide a 3\times6 gain matrix. In addition, users can specify saturation limits for the 
calculated relative MRP by specifying the ``relMRPMin`` and ``relMRPMax`` attributes.

Notably, the computed relative attitude is automatically combined with a reference attitude (provided either as another :ref:`AttRefMsgPayload`
or as the chief spacecrafts :ref:`NavAttMsgPayload`), allowing it to write an :ref:`AttRefMsgPayload` directly to a corresponding attitude control stack.

A simple example of this module's initialization alongside a recorder to store the reference attitude information is provided here:

.. code-block:: python
    :linenos:

    #       Configure a gain matrix; this one is for demonstration
    lqr_gain_set = np.array([[0,1,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0.25],
    [0,0,0],       ]).T #   Note that the gain matrix is 3x6, not 6x3

    #   Set up the hillStateConverter
    depAttRefData = hillToAttRef.hillToAttRef()
    depAttRefData.ModelTag = "dep_hillControl"
    depAttRefData.gainMatrix = hillToAttRef.MultiArray(lqr_gain_set)
    depAttRefData.hillStateInMsg.subscribeTo(hillStateMsg)
    if msg_type == 'NavAttMsg':
            depAttRefData.attNavInMsg.subscribeTo(attNavMsg)
    else:
            depAttRefData.attRefInMsg.subscribeTo(attRefMsg)

    if use_limits:
            depAttRefData.relMRPMin = -0.2 #    Configure minimum MRP
            depAttRefData.relMRPMax = 0.2  #    Configure maximum MRP

In addition, this module is used in the example script :ref:`scenarioDragRendezvous`, where it directly commands a spacecraft's attitude.
