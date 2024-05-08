Executive Summary
-----------------

This module reads in a message with an array of accelerometer measurements and integrates them to determine an accumulated :math:`\Delta\mathbf{v}` value.

On reset the net :math:`\Delta\mathbf{v}` is set to zero.  The output navigation message contains the latest measurements time tag and the total :math:`\Delta\mathbf{v}`. More information on can be found in the
:download:`PDF Description </../../src/fswAlgorithms/transDetermination/dvAccumulation/_Documentation/Basilisk-dvAccumulation-2019-03-28.pdf>`.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_dvAccumulation:
.. figure:: /../../src/fswAlgorithms/transDetermination/dvAccumulation/_Documentation/Images/moduleImgDvAccumulation.svg
    :align: center

    Figure 1: ``dvAccumulation()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - dvAcumOutMsg
      - :ref:`NavTransMsgPayload`
      - accumulated DV output message
    * - accPktInMsg
      - :ref:`AccDataMsgPayload`
      - input accelerometer message

