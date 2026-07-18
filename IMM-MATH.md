# IMM Mathematical Notes

This document gives the mathematical explanation of the trajectory prediction approach used by the project. It focuses on the Interacting Multiple Model filter, the role of model probabilities, the model transition matrix, and the reason IMM does not need an arbitrary fixed window of the last trajectory points.

## 1. Problem Statement

There is one moving object in three-dimensional space. At discrete measurement times

$$
t_1 < t_2 < \dots < t_k
$$

the system receives position measurements

$$
z_k \in \mathbb{R}^3.
$$

The measurement history available at step $k$ is

$$
Z_k = \{z_1,z_2,\dots,z_k\}.
$$

The goal is to solve two related Bayesian estimation problems.

Filtering:

$$
p(X_k \mid Z_k),
$$

which estimates the current hidden state after incorporating all measurements up to time $t_k$.

Prediction:

$$
p(X_{k+1} \mid Z_k), \qquad p(X_{k+h} \mid Z_k),
$$

which estimates a future state before future measurements arrive.

The hidden state is not directly observed. In the first production IMM formulation the common canonical state is

$$
X_k =
\begin{pmatrix}
p_x & p_y & p_z & v_x & v_y & v_z & a_x & a_y & a_z
\end{pmatrix}^T
\in \mathbb{R}^9.
$$

Here:

1. $(p_x,p_y,p_z)$ are position components;
2. $(v_x,v_y,v_z)$ are velocity components;
3. $(a_x,a_y,a_z)$ are acceleration components.

The canonical 9D state is important because IMM mixes states and covariances across models. Directly mixing a 6D constant-velocity state with a 9D constant-acceleration state is mathematically invalid. Therefore every model that participates in IMM must expose an adapter to and from the canonical space.

## 2. State-Space Model

The general nonlinear state-space model is

$$
X_k = f(X_{k-1},\Delta t_k) + w_k,
$$

$$
z_k = h(X_k) + v_k,
$$

where

$$
\Delta t_k = t_k - t_{k-1}.
$$

The variables are:

1. $X_k$: hidden state at step $k$;
2. $z_k$: measurement at step $k$;
3. $f(\cdot)$: motion model;
4. $h(\cdot)$: measurement model;
5. $w_k$: process noise;
6. $v_k$: measurement noise.

For Kalman-family filters the usual Gaussian assumptions are

$$
w_k \sim \mathcal{N}(0,Q_k),
$$

$$
v_k \sim \mathcal{N}(0,R_k).
$$

For a linear model:

$$
X_k = F_kX_{k-1}+w_k,
$$

$$
z_k = H_kX_k+v_k.
$$

Here $F_k$ is the state transition matrix and $H_k$ is the measurement matrix.

## 3. Single-Model Kalman Step

For a single linear Gaussian model, the Kalman filter performs prediction and correction.

All currently implemented Kalman-family filters in the backend use the canonical state

$$
X_k\in\mathbb{R}^9
$$

and the current Cartesian measurement

$$
z_k=
\begin{pmatrix}
x_k & y_k & z_k
\end{pmatrix}^T
\in\mathbb{R}^3.
$$

The implemented filters differ in how they propagate uncertainty, how they treat unexpectedly large innovations, and whether they use a linear transition matrix or local linearization.

### 3.1. Prediction

State prediction:

$$
\hat X_{k|k-1}=F_k\hat X_{k-1|k-1}.
$$

Covariance prediction:

$$
P_{k|k-1}=F_kP_{k-1|k-1}F_k^T+Q_k.
$$

Meaning:

1. the state estimate is propagated by the motion model;
2. uncertainty is propagated through the linear dynamics;
3. process noise $Q_k$ adds uncertainty caused by unmodeled behavior.

### 3.2. Correction

Innovation:

$$
\nu_k=z_k-H_k\hat X_{k|k-1}.
$$

Innovation covariance:

$$
S_k=H_kP_{k|k-1}H_k^T+R_k.
$$

Kalman gain:

$$
K_k=P_{k|k-1}H_k^TS_k^{-1}.
$$

State update:

$$
\hat X_{k|k}=\hat X_{k|k-1}+K_k\nu_k.
$$

Numerically stable Joseph covariance update:

$$
P_{k|k}
=
(I-K_kH_k)P_{k|k-1}(I-K_kH_k)^T
+K_kR_kK_k^T.
$$

The Joseph form is preferred because it helps preserve covariance symmetry and positive semi-definiteness.

### 3.3. Likelihood and Gating

For measurement dimension $m$, the Gaussian measurement likelihood is

$$
\Lambda_k
=
\frac{1}{\sqrt{(2\pi)^m|S_k|}}
\exp\left(
-\frac{1}{2}\nu_k^TS_k^{-1}\nu_k
\right).
$$

The squared Mahalanobis distance is

$$
d_k^2 = \nu_k^TS_k^{-1}\nu_k.
$$

If

$$
d_k^2 > \chi^2_{m,\alpha},
$$

then the measurement is suspicious under the current model and may be treated as an outlier.

For numerical stability the likelihood should be computed in log-domain:

$$
\log\Lambda_k
=
-\frac{1}{2}
\left[
m\log(2\pi)+\log|S_k|+\nu_k^TS_k^{-1}\nu_k
\right].
$$

### 3.4. State Initialization Used by the Implemented Filters

All implemented Kalman-family filters use the same canonical initialization rule. Let the initialization history be

$$
\{(t_i,z_i)\}_{i=1}^{n},
\qquad
t_1<t_2<\dots<t_n.
$$

The initial time is

$$
t_0=t_n.
$$

The initial position is the last measured Cartesian position:

$$
\hat p_{0|0}=z_n.
$$

If at least two measurements are available, the initial velocity is estimated by the last finite difference:

$$
\hat v_{0|0}
=
\frac{z_n-z_{n-1}}{t_n-t_{n-1}}.
$$

If at least three measurements are available, define

$$
\Delta t_1=t_{n-1}-t_{n-2},
\qquad
\Delta t_2=t_n-t_{n-1},
$$

$$
\hat v_1=\frac{z_{n-1}-z_{n-2}}{\Delta t_1},
\qquad
\hat v_2=\frac{z_n-z_{n-1}}{\Delta t_2}.
$$

Then the initial acceleration estimate is

$$
\hat a_{0|0}
=
\frac{\hat v_2-\hat v_1}
{\frac{1}{2}(\Delta t_1+\Delta t_2)}.
$$

If the history is too short to estimate velocity or acceleration, the corresponding canonical components are initialized by zero mean. The initial covariance is diagonal:

$$
P_{0|0}
=
\operatorname{diag}
\left(
\sigma_p^2,\sigma_p^2,\sigma_p^2,
\sigma_v^2,\sigma_v^2,\sigma_v^2,
\sigma_a^2,\sigma_a^2,\sigma_a^2
\right),
$$

where the implementation parameters are:

1. $\sigma_p^2$: initial position variance;
2. $\sigma_v^2$: initial velocity variance;
3. $\sigma_a^2$: initial acceleration variance.

This initialization is deliberately simple. Its role is to provide a valid canonical posterior seed for recursive filtering, not to solve a batch trajectory fitting problem.

### 3.5. Linear Kalman Filter

The `LinearKalmanFilter` is the standard linear Gaussian filter for a model pair

$$
X_k=F_kX_{k-1}+w_k,
\qquad
z_k=H_kX_k+v_k.
$$

The prediction is

$$
\hat X_{k|k-1}=F_k\hat X_{k-1|k-1},
$$

$$
P_{k|k-1}
=
F_kP_{k-1|k-1}F_k^T+Q_k.
$$

The predicted measurement mean is

$$
\hat z_{k|k-1}=H_k\hat X_{k|k-1},
$$

and its covariance is

$$
S_k
=
H_kP_{k|k-1}H_k^T+R_k.
$$

The innovation, likelihood, gating decision, Kalman gain, state update, and covariance update are exactly the equations in Sections 3.2 and 3.3.

In the current Cartesian implementation,

$$
H_k
=
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}.
$$

Therefore the measurement observes only position, while velocity and acceleration are inferred through the dynamic model and covariance coupling.

### 3.6. Fading-Memory Kalman Filter

The `FadingMemoryKalmanFilter` is a linear Kalman filter with covariance inflation during prediction. It uses the same state transition and measurement equations as the ordinary linear filter:

$$
X_k=F_kX_{k-1}+w_k,
\qquad
z_k=H_kX_k+v_k.
$$

Let

$$
\beta\ge 1
$$

be the fading factor. The state prediction is unchanged:

$$
\hat X_{k|k-1}=F_k\hat X_{k-1|k-1}.
$$

The covariance prediction becomes

$$
P_{k|k-1}
=
\beta F_kP_{k-1|k-1}F_k^T+Q_k.
$$

For $\beta=1$, this reduces to the ordinary linear Kalman filter. For $\beta>1$, old information is deliberately discounted because the prior covariance is inflated before the measurement update.

The predicted measurement covariance is still

$$
S_k
=
H_kP_{k|k-1}H_k^T+R_k,
$$

but $P_{k|k-1}$ is larger than in the ordinary filter when the propagated covariance is nonzero. Therefore:

1. the Kalman gain tends to place more weight on new measurements;
2. the Mahalanobis distance may become smaller for the same raw residual;
3. the filter adapts faster after maneuvers;
4. the estimate may become noisier if $\beta$ is too large.

The correction step is otherwise identical to the linear Kalman filter and uses Joseph covariance update.

### 3.7. Innovation-Adaptive Kalman Filter

The `InnovationAdaptiveKalmanFilter` is a linear Kalman filter that adapts the measurement noise used in the current correction step when the innovation is unexpectedly large.

First compute the ordinary predicted state and covariance:

$$
\hat X_{k|k-1}=F_k\hat X_{k-1|k-1},
$$

$$
P_{k|k-1}
=
F_kP_{k-1|k-1}F_k^T+Q_k.
$$

The raw innovation is

$$
\nu_k=z_k-H_k\hat X_{k|k-1}.
$$

Using the nominal measurement noise $R_k$, define the base innovation covariance

$$
S_{k,0}
=
H_kP_{k|k-1}H_k^T+R_k,
$$

and the base squared Mahalanobis distance

$$
d_{k,0}^2
=
\nu_k^TS_{k,0}^{-1}\nu_k.
$$

Let

$$
d_*^2>0
$$

be the target squared Mahalanobis distance, let

$$
\eta\in[0,1]
$$

be the adaptation rate, and let

$$
\gamma_{\max}\ge 1
$$

be the maximum measurement-noise scale. Define the innovation ratio

$$
\rho_k=\frac{d_{k,0}^2}{d_*^2}.
$$

The measurement-noise scale is

$$
\gamma_k
=
\begin{cases}
1,
&
\rho_k\le 1,
\\
\min\left(
\gamma_{\max},
1+\eta(\rho_k-1)
\right),
&
\rho_k>1.
\end{cases}
$$

The adapted measurement noise is

$$
\tilde R_k=\gamma_k R_k.
$$

The correction is then performed with

$$
\tilde S_k
=
H_kP_{k|k-1}H_k^T+\tilde R_k,
$$

$$
\tilde d_k^2
=
\nu_k^T\tilde S_k^{-1}\nu_k,
$$

$$
\log\tilde\Lambda_k
=
-\frac{1}{2}
\left[
m\log(2\pi)+\log|\tilde S_k|+\tilde d_k^2
\right].
$$

The adapted Kalman gain is

$$
\tilde K_k
=
P_{k|k-1}H_k^T\tilde S_k^{-1}.
$$

If the adapted gate accepts the measurement,

$$
\tilde d_k^2\le \chi^2_{m,\alpha},
$$

the update is

$$
\hat X_{k|k}
=
\hat X_{k|k-1}
+
\tilde K_k\nu_k,
$$

$$
P_{k|k}
=
(I-\tilde K_kH_k)
P_{k|k-1}
(I-\tilde K_kH_k)^T
+
\tilde K_k\tilde R_k\tilde K_k^T.
$$

If the measurement is rejected, the posterior estimate remains the prediction:

$$
\hat X_{k|k}=\hat X_{k|k-1},
\qquad
P_{k|k}=P_{k|k-1}.
$$

The adaptation is local to the correction step. It does not permanently rewrite the measurement model's nominal $R_k$. This makes the filter robust to isolated large residuals while preserving the configured sensor noise model for later steps.

### 3.8. Extended Kalman Filter

The `ExtendedKalmanFilter` is the implemented nonlinear Kalman-family filter. It supports nonlinear motion and nonlinear measurement models of the form

$$
X_k=f_k(X_{k-1})+w_k,
$$

$$
z_k=h_k(X_k)+v_k.
$$

The EKF replaces nonlinear propagation by first-order local linearization. At the previous posterior mean, define the motion Jacobian

$$
A_k
=
\left.
\frac{\partial f_k}{\partial X}
\right|_{\hat X_{k-1|k-1}}.
$$

The prediction is

$$
\hat X_{k|k-1}
=
f_k(\hat X_{k-1|k-1}),
$$

$$
P_{k|k-1}
=
A_kP_{k-1|k-1}A_k^T+Q_k.
$$

At the predicted state, define the measurement Jacobian

$$
H_k
=
\left.
\frac{\partial h_k}{\partial X}
\right|_{\hat X_{k|k-1}}.
$$

The predicted measurement is

$$
\hat z_{k|k-1}
=
h_k(\hat X_{k|k-1}),
$$

and the innovation is

$$
\nu_k
=
z_k-h_k(\hat X_{k|k-1}).
$$

The innovation covariance is

$$
S_k
=
H_kP_{k|k-1}H_k^T+R_k.
$$

The gain and update are

$$
K_k
=
P_{k|k-1}H_k^TS_k^{-1},
$$

$$
\hat X_{k|k}
=
\hat X_{k|k-1}+K_k\nu_k,
$$

$$
P_{k|k}
=
(I-K_kH_k)P_{k|k-1}(I-K_kH_k)^T
+K_kR_kK_k^T.
$$

The EKF uses the same log-likelihood and gating equations as the linear filter, but with the locally linearized $H_k$ and the nonlinear residual $z_k-h_k(\hat X_{k|k-1})$.

When both $f_k$ and $h_k$ are linear,

$$
f_k(X)=F_kX,
\qquad
h_k(X)=G_kX,
$$

the Jacobians are

$$
A_k=F_k,
\qquad
H_k=G_k,
$$

and the EKF equations reduce exactly to the ordinary linear Kalman filter equations.

## 4. Why IMM Is Needed

A single motion model encodes one hypothesis about the target dynamics. For example:

1. CV assumes approximately constant velocity;
2. CA assumes approximately constant acceleration;
3. Singer assumes time-correlated acceleration;
4. CT, in a future nonlinear version, assumes coordinated turning.

Real trajectories may switch between straight flight, acceleration, braking, turning, climbing, and abrupt maneuvering. A single Kalman filter can be excellent when its model matches the real dynamics, but it can become biased when the object switches to a different motion regime.

IMM addresses this by maintaining several filters simultaneously and treating the active model as a hidden discrete random variable.

## 5. IMM Definitions

Let the model set be

$$
\mathcal{M}=\{M_1,M_2,\dots,M_r\}.
$$

Each model has its own state estimate and covariance:

$$
\hat X_{k|k}^{(j)}, \qquad P_{k|k}^{(j)}.
$$

Each model also has a probability:

$$
\mu_k^{(j)} = P(M_j(k)\mid Z_k),
$$

with normalization

$$
\sum_{j=1}^{r}\mu_k^{(j)}=1.
$$

The full IMM posterior is approximated as a Gaussian mixture:

$$
p(X_k\mid Z_k)
\approx
\sum_{j=1}^r
\mu_k^{(j)}
\mathcal{N}
\left(
X_k;\hat X_{k|k}^{(j)},P_{k|k}^{(j)}
\right).
$$

The combined mean is a weighted sum:

$$
\hat X_{k|k}
=
\sum_{j=1}^r
\mu_k^{(j)}
\hat X_{k|k}^{(j)}.
$$

The combined covariance is not just the weighted covariance average. It must also include between-model spread:

$$
P_{k|k}
=
\sum_{j=1}^{r}
\mu_k^{(j)}
\left[
P_{k|k}^{(j)}
+
(\hat X_{k|k}^{(j)}-\hat X_{k|k})
(\hat X_{k|k}^{(j)}-\hat X_{k|k})^T
\right].
$$

## 6. Model Transition Matrix

IMM uses a model transition matrix

$$
\Pi =
\begin{pmatrix}
\pi_{11} & \pi_{12} & \cdots & \pi_{1r}\\
\pi_{21} & \pi_{22} & \cdots & \pi_{2r}\\
\vdots & \vdots & \ddots & \vdots\\
\pi_{r1} & \pi_{r2} & \cdots & \pi_{rr}
\end{pmatrix},
$$

where

$$
\pi_{ij}=P(M_j(k)\mid M_i(k-1)).
$$

The matrix is row-stochastic:

$$
\pi_{ij}\ge 0,
\qquad
\sum_{j=1}^{r}\pi_{ij}=1.
$$

The standard IMM assumption is that model evolution is a first-order Markov chain:

$$
P(M_k\mid M_{k-1},M_{k-2},\dots,Z_{k-1})
=
P(M_k\mid M_{k-1}).
$$

Therefore $\Pi$ is not normally recomputed from the last two or three measurements. It is an a priori model of how likely the target is to switch between motion regimes.

### 6.1. Typical Initial Values

If the model set is

$$
\mathcal{M}=\{CV,CA,Singer\},
$$

a reasonable transition matrix may be

$$
\Pi =
\begin{pmatrix}
0.95 & 0.04 & 0.01\\
0.03 & 0.94 & 0.03\\
0.02 & 0.08 & 0.90
\end{pmatrix}.
$$

Interpretation:

1. if the previous model was CV, the target is expected to remain CV with probability $0.95$;
2. if the previous model was CA, the target is expected to remain CA with probability $0.94$;
3. if the previous model was Singer, the target is expected to remain Singer with probability $0.90$.

Large diagonal entries encode persistence of motion regimes.

The approximate expected duration of model $M_j$ is

$$
\mathbb{E}[T_j]\approx \frac{1}{1-\pi_{jj}}.
$$

For example, if

$$
\pi_{CV,CV}=0.95,
$$

then

$$
\mathbb{E}[T_{CV}]\approx \frac{1}{1-0.95}=20
$$

steps.

### 6.2. Continuous-Time Construction

For nonuniform time steps it can be cleaner to define a continuous-time generator matrix $A$:

$$
A_{ij}=\lambda_{ij},\quad i\ne j,
$$

$$
A_{ii}=-\sum_{j\ne i}\lambda_{ij}.
$$

Then the discrete transition matrix for time interval $\Delta t$ is

$$
\Pi(\Delta t)=\exp(A\Delta t).
$$

This guarantees a valid stochastic transition matrix when $A$ is a valid Markov generator.

### 6.3. Does $\Pi$ Change Online?

In the standard IMM algorithm, $\Pi$ is fixed. What changes at every step is:

1. model probabilities $\mu_k^{(j)}$;
2. per-model estimates $\hat X_{k|k}^{(j)}$;
3. per-model covariances $P_{k|k}^{(j)}$.

Adaptive IMM variants may estimate $\Pi_k$ online, for example by using exponentially smoothed transition statistics:

$$
N_{ij,k}
=
\lambda N_{ij,k-1}
+
\xi_{ij,k},
$$

$$
\pi_{ij,k}
=
\frac{N_{ij,k}+\alpha_{ij}}
{\sum_{\ell=1}^{r}(N_{i\ell,k}+\alpha_{i\ell})}.
$$

Here:

1. $N_{ij,k}$ is an accumulated transition score;
2. $\lambda\in(0,1]$ is a forgetting factor;
3. $\alpha_{ij}$ is a prior pseudo-count;
4. $\xi_{ij,k}$ is an estimated soft transition responsibility.

This is a separate advanced extension. For the first implementation, a fixed and validated $\Pi$ is the more stable design.

## 7. Full IMM Algorithm

At step $k-1$, assume the filter state contains

$$
\left\{
\hat X_{k-1|k-1}^{(j)},
P_{k-1|k-1}^{(j)},
\mu_{k-1}^{(j)}
\right\}_{j=1}^{r}.
$$

### 1. Compute Predicted Model Probabilities

For each destination model $M_j$:

$$
c_j
=
\sum_{i=1}^{r}
\pi_{ij}\mu_{k-1}^{(i)}.
$$

Here $c_j$ is the prior probability of model $M_j$ before observing $z_k$.

### 2. Compute Mixing Probabilities

For each source model $M_i$ and destination model $M_j$:

$$
\mu_{k-1}^{(i|j)}
=
\frac{\pi_{ij}\mu_{k-1}^{(i)}}{c_j}.
$$

This is the probability that model $M_i$ was active at step $k-1$, conditional on considering model $M_j$ at step $k$.

### 3. Mix Initial State for Each Model

For each destination model $M_j$, form the mixed initial mean:

$$
\hat X_{k-1|k-1}^{0(j)}
=
\sum_{i=1}^{r}
\mu_{k-1}^{(i|j)}
\hat X_{k-1|k-1}^{(i)}.
$$

### 4. Mix Initial Covariance for Each Model

Define

$$
\Delta X_i^{(j)}
=
\hat X_{k-1|k-1}^{(i)}
-
\hat X_{k-1|k-1}^{0(j)}.
$$

Then

$$
P_{k-1|k-1}^{0(j)}
=
\sum_{i=1}^{r}
\mu_{k-1}^{(i|j)}
\left[
P_{k-1|k-1}^{(i)}
+
\Delta X_i^{(j)}(\Delta X_i^{(j)})^T
\right].
$$

This covariance accounts for both within-model uncertainty and disagreement between model means.

### 5. Predict Each Model

For each model $M_j$:

$$
\hat X_{k|k-1}^{(j)}
=
F_k^{(j)}
\hat X_{k-1|k-1}^{0(j)},
$$

$$
P_{k|k-1}^{(j)}
=
F_k^{(j)}
P_{k-1|k-1}^{0(j)}
(F_k^{(j)})^T
+
Q_k^{(j)}.
$$

For nonlinear models, $F_k^{(j)}$ is replaced by the appropriate EKF, UKF, CKF, or other nonlinear propagation mechanism.

### 6. Predict Measurement for Each Model

For each model:

$$
\hat z_{k|k-1}^{(j)}
=
h_j(\hat X_{k|k-1}^{(j)}).
$$

For a linear measurement model:

$$
\hat z_{k|k-1}^{(j)}
=
H_k^{(j)}\hat X_{k|k-1}^{(j)}.
$$

### 7. Produce the Prior Combined Prediction

Before observing $z_k$, the predicted measurement can be combined using $c_j$:

$$
\hat z_{k|k-1}
=
\sum_{j=1}^{r}
c_j\hat z_{k|k-1}^{(j)}.
$$

This is the IMM's prior prediction of the next measurement.

### 8. Receive the Actual Measurement

When the actual measurement $z_k$ arrives, compare it with each model's predicted measurement.

For each model:

$$
\nu_k^{(j)}
=
z_k-\hat z_{k|k-1}^{(j)}.
$$

### 9. Compute Innovation Covariance

For a linear measurement model:

$$
S_k^{(j)}
=
H_k^{(j)}
P_{k|k-1}^{(j)}
(H_k^{(j)})^T
+
R_k^{(j)}.
$$

### 10. Compute Mahalanobis Distance and Gating

For each model:

$$
d_{k,j}^2
=
(\nu_k^{(j)})^T
(S_k^{(j)})^{-1}
\nu_k^{(j)}.
$$

The measurement is consistent with model $M_j$ if

$$
d_{k,j}^2
\le
\chi^2_{m,\alpha}.
$$

### 11. Compute Model Log-Likelihoods

For each model:

$$
\log\Lambda_k^{(j)}
=
-\frac{1}{2}
\left[
m\log(2\pi)
+
\log|S_k^{(j)}|
+
d_{k,j}^2
\right].
$$

A smaller innovation and a reasonable covariance produce a larger likelihood.

### 12. Correct Each Model

For each model:

$$
K_k^{(j)}
=
P_{k|k-1}^{(j)}
(H_k^{(j)})^T
(S_k^{(j)})^{-1},
$$

$$
\hat X_{k|k}^{(j)}
=
\hat X_{k|k-1}^{(j)}
+
K_k^{(j)}\nu_k^{(j)}.
$$

The covariance is updated by Joseph form:

$$
P_{k|k}^{(j)}
=
(I-K_k^{(j)}H_k^{(j)})
P_{k|k-1}^{(j)}
(I-K_k^{(j)}H_k^{(j)})^T
+
K_k^{(j)}R_k^{(j)}(K_k^{(j)})^T.
$$

### 13. Update Model Probabilities

The standard probability update is

$$
\mu_k^{(j)}
=
\frac{\Lambda_k^{(j)}c_j}
{\sum_{\ell=1}^{r}\Lambda_k^{(\ell)}c_\ell}.
$$

In implementation this should be computed with log-sum-exp:

$$
a_j=\log\Lambda_k^{(j)}+\log c_j,
$$

$$
\log\mu_k^{(j)}
=
a_j-\operatorname{LSE}(a_1,\dots,a_r),
$$

where

$$
\operatorname{LSE}(a_1,\dots,a_r)
=
\log\left(\sum_{\ell=1}^{r}e^{a_\ell}\right).
$$

Then

$$
\mu_k^{(j)}=\exp(\log\mu_k^{(j)}).
$$

### 14. Combine the Updated State

The combined mean is

$$
\hat X_{k|k}
=
\sum_{j=1}^{r}
\mu_k^{(j)}
\hat X_{k|k}^{(j)}.
$$

The combined covariance is

$$
P_{k|k}
=
\sum_{j=1}^{r}
\mu_k^{(j)}
\left[
P_{k|k}^{(j)}
+
(\hat X_{k|k}^{(j)}-\hat X_{k|k})
(\hat X_{k|k}^{(j)}-\hat X_{k|k})^T
\right].
$$

### 15. Predict the Next Point

After the update, each model can be propagated to $t_{k+1}$:

$$
\hat X_{k+1|k}^{(j)}
=
f_j(\hat X_{k|k}^{(j)},\Delta t_{k+1}).
$$

The final predicted measurement is

$$
\hat z_{k+1|k}
=
\sum_{j=1}^{r}
\mu_k^{(j)}
h_j(\hat X_{k+1|k}^{(j)}).
$$

This is a weighted mean prediction. The full predictive distribution remains a mixture:

$$
p(X_{k+1}\mid Z_k)
=
\sum_{j=1}^{r}
\mu_k^{(j)}
p_j(X_{k+1}\mid Z_k,M_j).
$$

## 8. How IMM Decides Which Motion Model Is More Plausible

IMM does not use a rule such as "if the last three points are almost collinear, select CV." Instead, it updates model probabilities through likelihoods.

For two models, CV and CA:

$$
\log\frac{\mu_k^{CV}}{\mu_k^{CA}}
=
\log\frac{c_{CV}}{c_{CA}}
+
\log\Lambda_k^{CV}
-
\log\Lambda_k^{CA}.
$$

Using the Gaussian log-likelihood:

$$
\log\Lambda_k^{CV}-\log\Lambda_k^{CA}
=
-\frac{1}{2}
\left[
d_{CV}^2-d_{CA}^2
+
\log\frac{|S_{CV}|}{|S_{CA}|}
\right].
$$

Therefore a model gains probability when it explains the incoming measurement with a smaller normalized innovation and a consistent covariance.

The transition matrix $\Pi$ contributes inertia. A model does not instantly dominate after one lucky measurement unless the likelihood evidence is strong enough to overcome the prior transition probabilities.

## 9. How Much History Is Used?

IMM does not directly use a fixed number of last points and does not directly average over the full trajectory.

At step $k$, the relevant recursive statistics are

$$
\left\{
\hat X_{k|k}^{(j)},
P_{k|k}^{(j)},
\mu_k^{(j)}
\right\}_{j=1}^{r}.
$$

For a linear Gaussian filter, $(\hat X_{k|k},P_{k|k})$ is a sufficient recursive summary of the measurement history under that model. Past measurements are already compressed into:

1. the current estimated position;
2. the current estimated velocity;
3. the current estimated acceleration or model-specific dynamic state;
4. the uncertainty covariance;
5. the current model probabilities.

Therefore, if the trajectory already has 1000 complex points, IMM does not predict the next point by fitting a global average trajectory. Old maneuvers matter only through the current posterior state and covariance. The filter is fundamentally local in state, but recursive in information.

The effective memory is controlled by parameters:

$$
Q,\quad R,\quad \Pi,\quad P_0.
$$

Their qualitative effects are:

1. larger $Q$: faster adaptation, shorter dynamic memory, noisier estimates;
2. smaller $Q$: more inertia, smoother estimates, slower adaptation;
3. larger $R$: less trust in measurements;
4. smaller $R$: more trust in measurements;
5. larger $\pi_{jj}$: longer persistence of model $M_j$;
6. smaller $\pi_{jj}$: faster model switching;
7. larger initial $P_0$: less confidence in the initial state;
8. smaller initial $P_0$: stronger commitment to the initial state.

## 10. Is IMM Guaranteed to Predict Better Than a Single Kalman Filter?

No. IMM is not guaranteed to outperform the best single model on every trajectory.

If the real motion exactly matches one model and that model has well-tuned $Q$ and $R$, a single Kalman filter using that model can be equal to or better than IMM. IMM may be slightly worse because it mixes in other models and may switch with delay.

IMM is usually better when the trajectory contains changing motion regimes:

$$
\text{straight flight}
\rightarrow
\text{acceleration}
\rightarrow
\text{turn}
\rightarrow
\text{braking}.
$$

A single model is then structurally wrong on at least some segments, while IMM can redistribute probability among hypotheses.

The statement "IMM cannot be worse because it contains the single model" is false. If a good model has probability $0.7$ and poor models together have probability $0.3$, the weighted mean can be worse than the good model alone.

The practical interpretation is:

1. on a stable regime, the best single model may match or outperform IMM;
2. on an unknown trajectory with regime changes, IMM is typically more robust;
3. with a poor transition matrix $\Pi$, poor $Q/R$, or irrelevant models, IMM may be worse;
4. IMM reduces the risk of committing to one wrong motion model.

## 11. Practical Design Guidance for This Project

For the first implementation:

1. use a fixed row-stochastic $\Pi$;
2. choose large diagonal entries to encode regime persistence;
3. update $\mu_k^{(j)}$, not $\Pi$, at each online step;
4. compute model likelihoods in log-domain;
5. normalize model probabilities with log-sum-exp;
6. perform all IMM mixing in canonical 9D space;
7. expose model probabilities and gating diagnostics in update results;
8. tune $Q$, $R$, and $\Pi$ using simulator scenarios rather than hard-coded geometric rules such as "three points form a line."

The core idea is that IMM is a Bayesian filter over a hidden switching motion mode. It does not classify the whole historical trajectory. It recursively tracks the current state and current model probabilities, then predicts the next point as a probability-weighted mixture of model predictions.
