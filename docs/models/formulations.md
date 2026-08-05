# Model assumptions and mathematical formulations

## Hidden Markov model

An HMM assumes a discrete latent state \(S_t\in\{1,\ldots,K\}\), first-order homogeneous Markov
dynamics, and conditionally independent emissions:

\[
P(S_t=j\mid S_{1:t-1})=A_{S_{t-1},j},\qquad
p(y_{1:T},s_{1:T})=\pi_{s_1}p(y_1\mid s_1)
\prod_{t=2}^T A_{s_{t-1},s_t}p(y_t\mid s_t).
\]

A Gaussian emission uses \(Y_t\mid S_t=k\sim\mathcal N(\mu_k,\Sigma_k)\); a Student-t emission
adds heavier tails and degrees of freedom. Key assumptions include fixed \(K\), stable transitions
and emissions within the estimation window, correctly handled dependence, and sufficient visits to
each state. Return autocorrelation or covariate effects require an autoregressive or conditional
extension rather than wishful interpretation of a basic HMM.

Forward-backward inference costs approximately \(O(TK^2+TKD^2)\) with dense transitions and
full-covariance \(D\)-dimensional Gaussian emissions (covariance factorization can add
\(O(KD^3)\) per parameter update). Storage is \(O(TK)\) for smoothing and can be \(O(K)\) for a
streaming filter. EM multiplies fitting cost by its iterations and initializations.

## Markov-switching regression

One formulation is

\[
y_t=x_t^\top\beta_{S_t}+\varepsilon_t,\quad
\varepsilon_t\mid S_t=k\sim\mathcal N(0,\sigma_k^2).
\]

Intercepts, coefficients, and/or variance may switch. Exogenous regressors must be available at the
forecast timestamp. The same Markov, stability, and local-optimum concerns as an HMM apply.

## Mixtures and clustering

A finite mixture models \(p(y)=\sum_{k=1}^K w_kp(y\mid\theta_k)\), without sequential transition
dynamics. K-means minimizes \(\sum_t\lVert y_t-c_{z_t}\rVert^2\), implicitly favoring spherical,
equal-scale clusters under Euclidean distance. Both are usually retrospective unless embedded in a
strict fit-then-predict schedule. A Gaussian mixture EM iteration is roughly \(O(TKD^2)\) with full
covariances; K-means is roughly \(O(TKDI)\) for \(I\) iterations.

## Change-point objective

An offline segmentation commonly solves

\[
\min_{\tau_1<\cdots<\tau_m}
\sum_{r=0}^{m} C(y_{\tau_r+1:\tau_{r+1}})+\beta m,
\]

where \(C\) is a within-segment loss and \(\beta\) penalizes breaks. Binary segmentation is often
\(O(T\log T)\) for cheap costs; exact dynamic programming can be \(O(T^2)\); PELT is expected
linear under pruning conditions but has quadratic worst cases. CUSUM-style online detectors cost
about \(O(T)\) for a fixed univariate specification. Assumptions concern the changed parameter,
minimum segment length, independence/noise law, and abruptness of a break.

## Rules and supervised models

A threshold rule \(S_t=1[x_t>c]\) is transparent but assumes its feature, window, and cutoff remain
meaningful. A supervised classifier estimates \(P(L_t=k\mid X_t)\), where label quality replaces
latent-state assumptions; it learns the supplied label, not an independent “true regime.” Complexity
depends on the estimator. Report fit and inference time empirically along with sample size, feature
dimension, hardware, dependency version, iterations, and number of restarts.
