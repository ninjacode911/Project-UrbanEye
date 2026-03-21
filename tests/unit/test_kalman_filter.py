"""Tests for urbaneye.tracking.kalman_filter module."""

from __future__ import annotations

import numpy as np

from urbaneye.tracking.kalman_filter import KalmanFilter


class TestKalmanFilterInit:
    """Tests for Kalman Filter initialization."""

    def test_initiate_state_dimension(self) -> None:
        """Initiate creates 8D state vector."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)

    def test_initiate_covariance_dimension(self) -> None:
        """Initiate creates 8x8 covariance matrix."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, cov = kf.initiate(measurement)
        assert cov.shape == (8, 8)

    def test_initiate_position_matches_measurement(self) -> None:
        """Position components match the input measurement."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, _ = kf.initiate(measurement)
        np.testing.assert_array_almost_equal(mean[:4], measurement)

    def test_initiate_velocity_is_zero(self) -> None:
        """Initial velocity is zero (no prior motion information)."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, _ = kf.initiate(measurement)
        np.testing.assert_array_almost_equal(mean[4:], [0, 0, 0, 0])

    def test_initiate_covariance_is_positive_definite(self) -> None:
        """Covariance matrix has all positive eigenvalues."""
        kf = KalmanFilter()
        _, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)


class TestKalmanFilterPredict:
    """Tests for Kalman Filter prediction step."""

    def test_predict_moves_state_by_velocity(self) -> None:
        """Prediction moves position by the velocity component."""
        kf = KalmanFilter()
        mean = np.array([100, 200, 1.5, 50, 5, -3, 0, 0], dtype=np.float64)
        cov = np.eye(8) * 10
        new_mean, _ = kf.predict(mean, cov)
        # Position should shift by velocity
        assert new_mean[0] > mean[0]  # cx moved right (vx=5)
        assert new_mean[1] < mean[1]  # cy moved up (vy=-3)

    def test_predict_increases_uncertainty(self) -> None:
        """Prediction increases covariance (more uncertain without measurement)."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, cov = kf.initiate(measurement)
        _, new_cov = kf.predict(mean, cov)
        # Trace (sum of diagonal) should increase
        assert np.trace(new_cov) > np.trace(cov)

    def test_predict_preserves_symmetry(self) -> None:
        """Predicted covariance is symmetric."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        _, new_cov = kf.predict(mean, cov)
        np.testing.assert_array_almost_equal(new_cov, new_cov.T)


class TestKalmanFilterUpdate:
    """Tests for Kalman Filter update (correction) step."""

    def test_update_pulls_toward_measurement(self) -> None:
        """Update moves state closer to the measurement."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        mean, cov = kf.predict(mean, cov)

        # Measurement is offset from prediction
        measurement = np.array([110, 195, 1.5, 52])
        new_mean, _ = kf.update(mean, cov, measurement)

        # Updated state should be closer to measurement than prediction was
        dist_before = np.linalg.norm(mean[:4] - measurement)
        dist_after = np.linalg.norm(new_mean[:4] - measurement)
        assert dist_after < dist_before

    def test_update_reduces_uncertainty(self) -> None:
        """Update decreases covariance (measurement adds information)."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        mean, cov = kf.predict(mean, cov)
        _, new_cov = kf.update(mean, cov, np.array([100, 200, 1.5, 50]))
        assert np.trace(new_cov) < np.trace(cov)

    def test_predict_update_cycle_converges(self) -> None:
        """Repeated predict-update on constant-velocity track converges."""
        kf = KalmanFilter()
        measurement = np.array([100, 200, 1.5, 50])
        mean, cov = kf.initiate(measurement)

        # Simulate 20 frames of constant velocity motion
        for i in range(20):
            mean, cov = kf.predict(mean, cov)
            # Object moves right at 5px/frame
            true_pos = np.array([100 + (i + 1) * 5, 200, 1.5, 50])
            mean, cov = kf.update(mean, cov, true_pos)

        # After 20 updates, predicted velocity should approximate true velocity
        estimated_vx = mean[4]
        assert abs(estimated_vx - 5.0) < 1.0  # Within 1px/frame of true velocity


class TestKalmanFilterGating:
    """Tests for gating distance computation."""

    def test_close_measurement_low_distance(self) -> None:
        """Close measurement has low gating distance."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        close = np.array([[101, 201, 1.5, 50]])
        dist = kf.gating_distance(mean, cov, close)
        assert dist[0] < 100  # Should be small

    def test_far_measurement_high_distance(self) -> None:
        """Far measurement has high gating distance."""
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100, 200, 1.5, 50]))
        far = np.array([[500, 600, 3.0, 100]])
        dist = kf.gating_distance(mean, cov, far)
        assert dist[0] > 100  # Should be large
