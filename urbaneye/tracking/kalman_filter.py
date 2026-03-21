"""Linear Kalman Filter for bounding box tracking.

Implements the constant-velocity motion model used in ByteTrack/SORT:
  State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
  Measurement: [cx, cy, aspect_ratio, height]

The Kalman Filter predicts where each tracked object will be in the
next frame based on its estimated velocity, then corrects the prediction
using the actual detection (measurement).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilter:
    """Kalman Filter for bounding box state estimation.

    State vector (8D): [cx, cy, a, h, vx, vy, va, vh]
      - (cx, cy): bounding box center
      - a: aspect ratio (width / height)
      - h: height
      - (vx, vy, va, vh): respective velocities

    Measurement vector (4D): [cx, cy, a, h]
    """

    # Process noise scaling factors
    _std_weight_position: float = 1.0 / 20
    _std_weight_velocity: float = 1.0 / 160

    def __init__(self) -> None:
        """Initialize the Kalman Filter matrices."""
        ndim = 4  # measurement dimension

        # State transition matrix F (8x8): constant velocity model
        # x(t+1) = F @ x(t)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = 1.0  # position += velocity

        # Measurement matrix H (4x8): extract position from state
        # z = H @ x
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create track state from unassociated measurement.

        Args:
            measurement: Bounding box [cx, cy, a, h].

        Returns:
            Tuple of (mean, covariance) for the new track.
            mean: 8D state vector.
            covariance: 8x8 covariance matrix.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        # Initial covariance: high uncertainty for velocity
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman Filter prediction step.

        Projects the state and covariance forward by one time step
        using the constant-velocity motion model.

        Args:
            mean: Current 8D state vector.
            covariance: Current 8x8 covariance matrix.

        Returns:
            Tuple of (predicted_mean, predicted_covariance).
        """
        # Process noise Q
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        # Predict: x' = F @ x, P' = F @ P @ F^T + Q
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state to measurement space.

        Args:
            mean: 8D state vector.
            covariance: 8x8 covariance matrix.

        Returns:
            Tuple of (projected_mean, projected_covariance) in measurement space.
        """
        # Measurement noise R
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        # Project: z' = H @ x, S = H @ P @ H^T + R
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman Filter correction step.

        Updates the state using a new measurement (detection).

        Args:
            mean: Predicted 8D state vector.
            covariance: Predicted 8x8 covariance matrix.
            measurement: Observed bounding box [cx, cy, a, h].

        Returns:
            Tuple of (corrected_mean, corrected_covariance).
        """
        # Project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain: K = P @ H^T @ S^{-1}
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
        ).T

        # Innovation: y = z - H @ x
        innovation = measurement - projected_mean

        # Update: x = x + K @ y, P = (I - K @ H) @ P
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute Mahalanobis distance for measurement-track gating.

        Args:
            mean: 8D state vector.
            covariance: 8x8 covariance matrix.
            measurements: Nx4 measurement array.
            only_position: If True, only use cx, cy for distance.

        Returns:
            Array of squared Mahalanobis distances (N,).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        chol = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True)
        return np.sum(z * z, axis=0)
