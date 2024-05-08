Executive Summary
-----------------

This module reads in the position and velocity of multiple orbital bodies and outputs position and velocity of each body relative to a single other orbital body position and velocity.  Up to 10 input ephemeris messages can be connected.

More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/transDetermination/ephemDifference/_Documentation/Basilisk-ephemDifference-2019-03-27.pdf>`.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_ephemDifference:
.. figure:: /../../src/fswAlgorithms/transDetermination/ephemDifference/_Documentation/Images/moduleImgEphemDifference.svg
    :align: center

    Figure 1: ``ephemDifference()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - ephBaseInMsg
      - :ref:`EphemerisMsgPayload`
      - base ephemeris input message name
    * - ephInMsg
      - :ref:`EphemerisMsgPayload`
      - ephemeris input message to be converted, stored in ``changeBodies[i]``
    * - ephOutMsg
      - :ref:`EphemerisMsgPayload`
      - converted ephemeris output message, stored in ``changeBodies[i]``

