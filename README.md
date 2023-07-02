# AttitudeEstimation-IEKF
This repository provides a simple implementation of the Invariant Extended Kalman Filter (IEKF) for attitude estimation. It focuses on estimating the orientation using observations of two vectors with known directions in the fixed frame. The code was developed from scratch due to the lack of existing straightforward implementations available online. It serves as a valuable resource for those in search of a simple IEKF solution for attitude estimation that may not be readily accessible elsewhere on the internet (to the best of the author's knowledge).

For the mathematics behind and algorithm explaination, I recomend to read this paper https://www.sciencedirect.com/science/article/pii/S2405896317300782.

## Mathematical model
### dynamic model:
$$R_{k+1}=R_k \exp_m(\omega_k dt+w_k),$$
where:
- $R_k$ represents the rotation matrix at time step $k$, this rotation from body frame to the fixed frame ($R_B^N$).
- $\exp_m$ refers to the exponential map, which maps elements from $\mathbb{R}^3$ to elements in the Lie algebra $SO(3)$.
- $\omega_k$ denotes the angular velocity at time step $k$.
- $w_k$ represents the process noise, assumed to follow a Gaussian distribution with a mean of zero and a covariance matrix $Q_k$.
- $dt$ sampling time.
  
To gain a better understanding of the exponential map and the fundamentals related to Lie groups, I recommend reading: https://www.iri.upc.edu/files/scidoc/2089-A-micro-Lie-theory-for-state-estimation-in-robotics.pdf.
### output model (observation function):
$$(V_k^{1B},V_k^{2B})=(R_k^T V_k^{1N}+v_k^1,R_k^T V_k^{2N}+v_k^2),$$
where:
- $R_k^T$ is the rotation matrix from navigation (fixed) frame to the body frame ($R_N^B=(R_B^N)^{-1}=(R_B^N)^{T}$).
- $V_k^{1N}$, $V_k^{2N}$ are two known vectors in the navigation (fixed) frame.
- $V_k^{1B}$, $V_k^{2B}$ are the measured vectors.
- $v_k^1$ and $v_k^2$ are the measurement noises, assumed to follow a Gaussian distribution with a mean of zero and known covarianc matrix.
