Executive Summary
-----------------

Converter that takes an ephemeris output message and converts it over to a translational state estimate message.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. _ModuleIO_ephemNavConverter:
.. figure:: /../../src/fswAlgorithms/transDetermination/ephemNavConverter/_Documentation/Images/moduleImgEphemNavConverter.svg
    :align: center

    Figure 1: ``ephemNavConverter()`` Module I/O Illustration

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - ephInMsg
      - :ref:`EphemerisMsgPayload`
      - ephemeris input message
    * - stateOutMsg
      - :ref:`NavTransMsgPayload`
      - navigation output message


