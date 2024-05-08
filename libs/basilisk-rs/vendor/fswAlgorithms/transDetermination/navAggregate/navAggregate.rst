Executive Summary
-----------------

This module takes in a series of navigation messages and constructs a navigation output message using a select subset of information from the input messages.  For more information see the
:download:`PDF Description </../../src/fswAlgorithms/transDetermination/navAggregate/_Documentation/Basilisk-navAggregate-2019-02-21.pdf>`.


Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_navAggregate:
.. figure:: /../../src/fswAlgorithms/transDetermination/navAggregate/_Documentation/Images/moduleImgNavAggregate.svg
    :align: center

    Figure 1: ``navAggregate()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - navAttOutMsg
      - :ref:`NavAttMsgPayload`
      - blended attitude navigation output message
    * - navTransOutMsg
      - :ref:`NavTransMsgPayload`
      - blended translation navigation output message
    * - navAttInMsg
      - :ref:`NavAttMsgPayload`
      - attitude navigation input message stored inside the ``AggregateAttInput`` structure
    * - navTransInMsg
      - :ref:`NavTransMsgPayload`
      - translation navigation input message stored inside the ``AggregateTransInput`` structure

