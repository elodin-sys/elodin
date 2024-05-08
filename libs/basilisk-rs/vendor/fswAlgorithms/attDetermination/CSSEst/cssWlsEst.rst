Executive Summary
-----------------
A weighted least-squares minimum-norm algorithm is used to estimate the body-relative sun heading using a cluster of coarse sun sensors.  Using two successive sun heading evaluation the module also computes the inertial angular velocity vector.  As rotations about the sun-heading vector are not observable, this angular velocity vector only contains body rates orthogonal to this sun heading vector.  More information on can be found in the
:download:`PDF Description </../../src/fswAlgorithms/attDetermination/CSSEst/_Documentation/Basilisk-cssWlsEst-20180429.pdf>`

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  The module msg connection is set by the
user from python.  The msg type contains a link to the message structure definition, while the description
provides information on what this message is used for.

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - cssDataInMsg
      - :ref:`CSSArraySensorMsgPayload`
      - name of the CSS sensor input message
    * - cssConfigInMsg
      - :ref:`CSSConfigMsgPayload`
      - name of the CSS configuration input message
    * - navStateOutMsg
      - :ref:`NavAttMsgPayload`
      - name of the navigation output message containing the estimated states
    * - cssWLSFiltResOutMsg
      - :ref:`SunlineFilterMsgPayload`
      - name of the CSS filter state data out message
