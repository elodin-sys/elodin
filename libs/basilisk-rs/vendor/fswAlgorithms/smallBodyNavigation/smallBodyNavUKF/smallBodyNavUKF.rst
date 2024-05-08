Executive Summary
-----------------
This module provides a navigation solution for a spacecraft about a small body. An unscented Kalman filter estimates relative spacecraft position and velocity with respect to the small body, and the non-Keplerian acceleration term.

This module is only meant to test the possibility of estimating pairs of position and acceleration which may be post processed to estimate the small body gravity field. Therefore, realistic measurement modules do not exist to support this module, and not every source of uncertainty in the problem is an estimated parameter. Future work will build upon this filter.

Message Connection Descriptions
-------------------------------
The following table lists all the module input and output messages.  
The module msg connection is set by the user from python.  
The msg type contains a link to the message structure definition, while the description 
provides information on what this message is used for.

.. _ModuleIO_smallBodyNavUKF:
.. figure:: /../../src/fswAlgorithms/smallBodyNavigation/smallBodyNavUKF/_Documentation/Images/moduleIOSmallBodyNavigationUKF.svg
    :align: center

    Figure 1: ``smallBodyNavUKF()`` Module I/O Illustration

Note that this C++ FSW module provides both C- and C++-wrapped output messages.  The regular C++ wrapped output messages end with the usual ``...OutMsg``.  The C wrapped output messages have the same payload type, but end with ``...OutMsgC``.  

.. list-table:: Module I/O Messages
    :widths: 25 25 50
    :header-rows: 1

    * - Msg Variable Name
      - Msg Type
      - Description
    * - navTransInMsg
      - :ref:`NavTransMsgPayload`
      - Translational nav input message
    * - asteroidEphemerisInMsg
      - :ref:`EphemerisMsgPayload`
      - Small body ephemeris input message
    * - smallBodyNavUKFOutMsg
      - :ref:`SmallBodyNavUKFMsgPayload`
      - Small body nav output msg - states and covariances
    * - smallBodyNavUKFOutMsgC
      - :ref:`SmallBodyNavUKFMsgPayload`
      - C-wrapped small body nav output msg - states and covariances


Detailed Module Description
---------------------------
General Function
^^^^^^^^^^^^^^^^
The ``smallBodyNavUKF()`` module provides a complete translational state estimate for a spacecraft in proximity of a small body. 
The relative spacecraft position and velocity, and the non-Keplerian acceleration (e.g. inhomogeneous gravity field) are estimated 
by the filter. The filter assumes full observability of the relative position and that the small body state is perfectly known. 
Future work may consider augmenting the estimates and using more realistic measurements. The full state vector may be found below:

.. math::
    :label: eq:state

    \mathbf{x} =
    \begin{bmatrix}
    {}^A\mathbf{r}_{S/A} \\
    {}^A\mathbf{v}_{S/A} \\
    {}^A\mathbf{a} \\
    \end{bmatrix}

The associated frame definitions may be found in the following table.

.. list-table:: Frame Definitions
    :widths: 25 25
    :header-rows: 1

    * - Frame Description
      - Frame Definition
    * - Small Body Fixed Frame
      - :math:`A: \{\hat{\mathbf{a}}_1, \hat{\mathbf{a}}_2, \hat{\mathbf{a}}_3\}`
    * - J2000 Inertial Frame
      - :math:`N: \{\hat{\mathbf{n}}_1, \hat{\mathbf{n}}_2, \hat{\mathbf{n}}_3\}`

Initialization
^^^^^^^^^^^^^^

The module start by initializing the weights for the unscented transform

.. math::
    :label: eq:UT_weights

    w^{[0]}_{m}=\kappa / (\kappa + N),\>\>\>\>\>\> w^{[0]}_{c}=w^{[0]}_{m}+1-\alpha^2+\beta,\>\>\>\>\>\> w^{[i]}_{m}=w^{[i]}_{c}=1/(2N+\kappa) \>\> i\neq 0
 

Algorithm
^^^^^^^^^^
This module employs an unscented Kalman filter (UKF) `Wan and Van Der Merwe <https://doi.org/10.1109/ASSPCC.2000.882463>`__ to estimate the
relevant states. The UKF relies on the unscented transform (UT) to compute the non-linear transformation of a Gaussian distribution. Let 
consider a random variable :math:`\mathbf{x}` of dimension :math:`N` modelled as a Gaussian distribution with mean :math:`\hat{\mathbf{x}}` 
and covariance :math:`P`. The UT computes numerically the resulting mean and covariance of :math:`\mathbf{y}=\mathbf{f}(\mathbf{x})` by
creating :math:`2N+1` samples named sigma points as

.. math::
    :label: eq:sigma_points

    \pmb{\chi}^{[i]} = \hat{\mathbf{x}} \pm \left(\sqrt{(N+\kappa) P}\right)_{|i|},\>\> i = -N...N

where :math:`|i|` denotes the columns of the matrix. Then, transform each sigma point as :math:`\pmb{\xi}^{[i]}=\mathbf{f}(\pmb{\chi}^{[i]})`. Finally, compute the mean and covariance of 
:math:`\mathbf{y}=\mathbf{f}(\mathbf{x})` as

.. math::
    :label: eq:mean_sigma

    \hat{\mathbf{y}} = \sum^{N}_{i=-N}w^{[i]}_{m}\pmb{\xi}^{[i]}

.. math::
    :label: eq:covar_sigma

    R = \sum^{N}_{i=-N}w^{[i]}_{c}(\pmb{\xi}^{[i]} - \hat{\mathbf{y}})(\pmb{\xi}^{[i]} - \hat{\mathbf{y}})^T

In the small body scenario under consideration, there are two transformation functions. The process propagation, assumed as simple 
forward Euler integration, as

.. math::
    :label: eq:process_propagation

    \mathbf{x}_{k+1}=\Delta t
    \begin{bmatrix}
    0_{3\times3} & I & 0_{3\times3}\\
    -{}^A\Omega^2_{A/N} & -2{}^A\Omega_{A/N} & I\\
    0_{3\times3} & 0_{3\times3} & 0_{3\times3}\\
    \end{bmatrix}\mathbf{x}_{k}+
    \Delta t
    \begin{bmatrix}
    0_{3\times1}\\
    -\mu \mathbf{r}_k / ||\mathbf{r}_k||^3 \\
    0_{3\times1}\\
    \end{bmatrix}

Note that :math:`{}^A\Omega_{A/N}` is the cross-product matrix associated to the small body rotation rate. The state to measurements transformation is

.. math::
    :label: eq:meas_transform

    \mathbf{y}_{k+1}=\begin{bmatrix}
    I & 0_{3\times3} & 0_{3\times3}\\
    \end{bmatrix}\mathbf{x}_{k+1}

Under the previous considerations, the UKF estimation is as follows:

1) Compute the a-priori state estimation :math:`\hat{\mathbf{x}}^{-}_{k+1}` and :math:`P^{-}_{k+1}` 
through the UT to the propagation function. Add the process noise uncertainty :math:`P_{proc}`

.. math::
    :label: eq:process_noise_addition

    P^{-}_{k+1}\leftarrow P^{-}_{k+1} + P_{proc}

2) Compute the a-priori measurements :math:`\hat{\mathbf{y}}^{-}_{k+1}` and :math:`R^{-}_{k+1}` through
the UT to the state to measurements transformation function. Add the measurements uncertainty :math`R_{meas}`

.. math::
    :label: eq:measurements_noise_addition

    R^{-}_{k+1}\leftarrow R^{-}_{k+1} + R_{meas}

3) Compute the cross-correlation matrix between state and measurements and the Kalman gain

.. math::
    :label: eq:cross_correlation

    H = \sum^{N}_{i=-N}w^{[i]}_{c}(\pmb{\chi}^{[i]} - \hat{\mathbf{x}}^{-}_{k+1})(\pmb{\xi}^{[i]} - \hat{\mathbf{y}}^{-}_{k+1})^T

.. math::
    :label: eq:kalman_gain_UKF

    K = H (R^{-}_{k+1})^{-1}

4) Update the state estimation with the incoming measurement

.. math::
    :label: eq:kalman_update_mean

    \mathbf{x}_{k+1} = \mathbf{x}^{-}_{k+1} + K(\mathbf{y}_{k+1} - \hat{\mathbf{y}}^{-}_{k+1})

.. math::
    :label: eq:kalman_update_covar

    P_{k+1} = P^{-}_{k+1} - KR^{-}_{k+1}K^T
	

These steps are based on `Wan and Van Der Merwe <https://doi.org/10.1109/ASSPCC.2000.882463>`__ (see algorithm 3.1). The weights selection
can be consulted there but it is the one described in the initialization step. The filter hyper-parameters are :math:`\{\alpha, \beta, \kappa\}`.
Note that :math:`\kappa` is equivalent to :math:`\lambda` in the original publication.


Module Assumptions and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module assumptions and limitations are listed below:

 - The reference frame within the filter is rigidly attached to the small body
 - The filter assumes the small body mass and rotational state are perfectly known
 - The small body has a uniform rotation around its principal inertia axis
 - Currently, the prediction and update rates occur at the same frequency
 - The position measurement are produced by simpleNavMeas, thus being referred to the J2000 inertial frame. The position is internally translated to the asteroid fixed frame coordinates.


User Guide
^^^^^^^^^^
The user then must set the following module variables:

- ``mu_ast``, the gravitational constant of the small body in :math:`\text{m}^3/\text{s}^2`
- ``P_proc``, the process noise covariances
- ``R_meas``, the measurement noise covariance
- ``x_hat_k`` to initialize the state estimation
- ``P_k`` to initialize the state estimation covariance

The user could opt to set the following module variables (initialized by default):

- ``alpha``, filter hyper-parameter (2 by default)
- ``beta``, filter hyper-parameter (0 by default)
- ``kappa``, filter hyper-parameter (:math:`10^{-3}` by default)

The user must connect to each input message described in Table 1. 