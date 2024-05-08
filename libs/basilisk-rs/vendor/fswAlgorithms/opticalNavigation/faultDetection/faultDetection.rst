Executive Summary
-----------------
This module is a fault detection module for optical navigation. It uses a scenario in which two image processing methods
are implemented, and then can be compared in different ways.

Module Assumptions and Limitations
----------------------------------
There are no direct assumptions in this module. The performance and limitations are tied to the two methods that are
input.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg variable name is set by the user from python.  The msg type contains a link to the message structure definition, while the description provides information on what this message is used for.


.. table:: Module I/O Messages
        :widths: 25 25 100

        +------------------------+---------------------------------+---------------------------------------------------+
        | Msg Variable Name      | Msg Type                        | Description                                       |
        +========================+=================================+===================================================+
        | navMeasPrimaryInMsg    | :ref:`OpNavMsgPayload`          | Input primary nav message                         |
        +------------------------+---------------------------------+---------------------------------------------------+
        | navMeasSecondaryInMsg  | :ref:`OpNavMsgPayload`          | Input secondary nav message                       |
        +------------------------+---------------------------------+---------------------------------------------------+
        | cameraConfigInMsg      | :ref:`CameraConfigMsgPayload`   | Input camera message                              |
        +------------------------+---------------------------------+---------------------------------------------------+
        | attInMsg               | :ref:`NavAttMsgPayload`         | Input attitude message                            |
        +------------------------+---------------------------------+---------------------------------------------------+
        | opNavOutMsg            | :ref:`OpNavMsgPayload`          | Ouput navigation message given the two inputs     |
        +------------------------+---------------------------------+---------------------------------------------------+


Detailed Module Description
---------------------------

The document provides details that are reminded here:
Three fault modes are possible:

- ``FaultMode`` = 0: is the less restrictive: it uses either of the measurements available and merges them if they are both available

- ``FaultMode`` = 1: is more restrictive: only the primary is used if both are available and the secondary is only used for a dissimilar check

- ``FaultMode`` = 2: is most restrictive: the primary is not used in the absence of the secondary measurement

Equations
^^^^^^^^^
The important merging equations for the state and covariance of the two inputs are given here

.. math::
    P = (P1^{-1} + P2^{-1})^{-1}
    x = P (P1^{-1}x1 + P2^{-1}x2)

The rest of the module is logic driven as explained in the doxygen documentation.

User Guide
----------
An example setup is provided here:

.. code-block:: python
    :linenos:

    faults = faultDetection.FaultDetectionData()
    faults.sigmaFault = 3
    faults.faultMode = 1

The sigmaFault parameter is the multiplier on the covariances that needs to be passed for the faults to be triggered
