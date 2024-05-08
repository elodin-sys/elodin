Executive Summary
-----------------

This module takes the TDB time, current object time and computes the state of the object using the time corrected by TDB and the stored Chebyshev coefficients.

If the time provided is outside the specified range for which the stored Chebyshev coefficients are valid then the position vectors rail high/low appropriately.  More information can be found in the
:download:`PDF Description </../../src/fswAlgorithms/transDetermination/oeStateEphem/_Documentation/Basilisk-oeStateEphem-20190426.pdf>`.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_oeStateEphem:
.. figure:: /../../src/fswAlgorithms/transDetermination/oeStateEphem/_Documentation/Images/moduleImgOeStateEphem.svg
    :align: center

    Figure 1: ``oeStateEphem()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - stateFitOutMsg
      - :ref:`EphemerisMsgPayload`
      - output navigation message for pos/vel
    * - clockCorrInMsg
      - :ref:`TDBVehicleClockCorrelationMsgPayload`
      - clock correlation input message

