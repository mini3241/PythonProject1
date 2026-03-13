"""
Kalman filter tracker - imported from existing codebase.
Preserves the original tracking logic.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


class FusionState(IntEnum):
    """Detection source state."""
    NONE = 0
    CAMERA_ONLY = 1
    RADAR_ONLY = 2
    FUSED = 3


@dataclass
class Detection:
    """Detection result data class."""
    center: Tuple[float, float]
    confidence: float
    feature: Optional[np.ndarray] = None
    fusion_state: FusionState = FusionState.RADAR_ONLY


class KalmanFilterFusion:
    """Kalman filter for 2D tracking with orientation."""

    def __init__(self, dt=1.0):
        self.dt = dt
        self.dim_x = 6  # [x, y, theta, vx, vy, vtheta]
        self.dim_z = 3  # [x, y, theta]

        # Motion model
        self._motion_mat = np.eye(self.dim_x, dtype=np.float32)
        self._motion_mat[0, 3] = dt
        self._motion_mat[1, 4] = dt
        self._motion_mat[2, 5] = dt

        # Observation model
        self.H = np.eye(self.dim_z, self.dim_x, dtype=np.float32)

        # Process noise
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._std_weight_orientation = 1.0 / 20
        self._std_weight_orientation_velocity = 1.0 / 160

        q = [
            self._std_weight_position, self._std_weight_position, self._std_weight_orientation,
            self._std_weight_velocity, self._std_weight_velocity, self._std_weight_orientation_velocity
        ]
        self.Q = np.diag(np.square(q)).astype(np.float32)

        # Measurement noise
        r = [self._std_weight_position, self._std_weight_position, self._std_weight_orientation]
        self.R = np.diag(np.square(r)).astype(np.float32)

        self.mean = None
        self.covariance = None

    def initiate(self, x, y, orientation=0.0):
        self.mean = np.array([x, y, orientation, 0, 0, 0], dtype=np.float32).reshape(self.dim_x, 1)
        std = [
            self._std_weight_position, self._std_weight_position, self._std_weight_orientation,
            self._std_weight_velocity, self._std_weight_velocity,
            self._std_weight_orientation_velocity
        ]
        self.covariance = np.diag(np.square(std)).astype(np.float32)
        return self.mean.flatten(), self.covariance

    def predict(self):
        self.mean = self._motion_mat @ self.mean
        self.covariance = self._motion_mat @ self.covariance @ self._motion_mat.T + self.Q
        return self.mean.flatten(), self.covariance

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32).reshape(self.dim_z, 1)
        PHT = self.covariance @ self.H.T
        S = self.H @ PHT + self.R
        K = PHT @ np.linalg.inv(S)
        innovation = z - self.H @ self.mean
        self.mean = self.mean + K @ innovation
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.covariance = I_KH @ self.covariance
        return self.mean.flatten(), self.covariance


def speed_direction(pre_xy, curr_xy):
    x1, y1 = pre_xy[0], pre_xy[1]
    x2, y2 = curr_xy[0], curr_xy[1]
    speed = np.array([y2 - y1, x2 - x1])
    norm = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-6
    return speed / norm


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return np.array([-100, -100, -1], dtype=np.float32)
    for i in range(k):
        dt = cur_age - i - 1
        if dt in observations:
            return observations[dt]
    max_age = max(observations.keys())
    return observations[max_age]


class KalmanTrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class KalmanTrack:
    """Kalman track with appearance features."""
    _next_id = 1

    def __init__(self, x, y, n_init=2, max_age=5, delta_t=3, feature=None):
        self.kf = KalmanFilterFusion()
        self.mean, self.covariance = self.kf.initiate(x, y, orientation=0.0)
        self.track_id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.delta_t = delta_t
        self.velocity = None
        self.track_orientation = 0.0
        self.last_observation = np.array([-100, -100, -1], dtype=np.float32)
        self.observations = {}
        self.state = KalmanTrackState.Confirmed if self.hits >= self.n_init else KalmanTrackState.Tentative

        self.feature = feature if feature is not None else None
        self.feature_history = []
        self.fusion_state = FusionState.RADAR_ONLY

    @property
    def position(self):
        return self.mean[:2]

    @property
    def orientation(self):
        return self.mean[2] if len(self.mean) > 2 else 0.0

    def predict(self):
        self.mean, self.covariance = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, x, y, feature=None, fusion_state=FusionState.RADAR_ONLY):
        self.mean, self.covariance = self.kf.update([x, y, 0.0])
        self.hits += 1
        self.time_since_update = 0

        prev_obs = k_previous_obs(self.observations, self.age, self.delta_t)
        if prev_obs[2] >= 0:
            self.velocity = speed_direction(prev_obs[:2], [x, y])

        self.last_observation = np.array([x, y, self.age], dtype=np.float32)
        self.observations[self.age] = self.last_observation

        if self.state == KalmanTrackState.Tentative and self.hits >= self.n_init:
            self.state = KalmanTrackState.Confirmed

        if feature is not None and np.any(feature):
            self.feature = feature
            self.feature_history.append(feature)
            if len(self.feature_history) > 10:
                self.feature_history.pop(0)

        self.fusion_state = fusion_state

    def mark_missed(self):
        if self.state == KalmanTrackState.Tentative:
            self.state = KalmanTrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = KalmanTrackState.Deleted


def hungarian_algorithm(cost_matrix):
    """Simple Hungarian algorithm implementation."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


class SequenceMOTATracker:
    """Multi-object tracker with Kalman filter and appearance features."""

    def __init__(self, distance_threshold_primary=3.0, distance_threshold_secondary=5.0,
                 vdc_weight=10.0, app_weight=0.5, use_appearance=True):
        self.tracks = []
        self.frame_count = 0
        self.DISTANCE_THRESHOLD_PRIMARY = distance_threshold_primary
        self.DISTANCE_THRESHOLD_SECONDARY = distance_threshold_secondary
        self.VDC_WEIGHT = vdc_weight
        self.APP_WEIGHT = app_weight
        self.USE_APPEARANCE = use_appearance

    def predict(self):
        for track in self.tracks:
            track.predict()

    def _get_distance_cost_matrix(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                if isinstance(det, Detection):
                    det_x, det_y = det.center[0], det.center[1]
                else:
                    det_x, det_y = det[0], det[1]
                dist = np.sqrt((track.position[0] - det_x) ** 2 + (track.position[1] - det_y) ** 2)
                cost_matrix[t, d] = dist
        return cost_matrix

    def _get_orientation_cost_matrix(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, track in enumerate(self.tracks):
            if track.velocity is None:
                continue
            prev_obs = k_previous_obs(track.observations, track.age, track.delta_t)
            if prev_obs[2] < 0:
                continue
            for d, det in enumerate(detections):
                if isinstance(det, Detection):
                    det_x, det_y = det.center[0], det.center[1]
                else:
                    det_x, det_y = det[0], det[1]
                actual_dy = det_y - prev_obs[1]
                actual_dx = det_x - prev_obs[0]
                actual_norm = np.sqrt(actual_dy ** 2 + actual_dx ** 2) + 1e-6
                actual_direction = np.array([actual_dy / actual_norm, actual_dx / actual_norm])
                cos_sim = np.dot(track.velocity, actual_direction)
                cost_matrix[t, d] = (1 - cos_sim) * self.VDC_WEIGHT
        return cost_matrix

    def _get_appearance_cost_matrix(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

        if not self.USE_APPEARANCE:
            return cost_matrix

        for t, track in enumerate(self.tracks):
            if track.feature is None:
                continue
            track_feat = track.feature
            for d, det in enumerate(detections):
                if isinstance(det, Detection) and det.feature is not None and np.any(det.feature):
                    det_feat = det.feature
                    norm_t = np.linalg.norm(track_feat)
                    norm_d = np.linalg.norm(det_feat)
                    if norm_t > 1e-6 and norm_d > 1e-6:
                        cos_sim = np.dot(track_feat, det_feat) / (norm_t * norm_d)
                        cost_matrix[t, d] = (1 - cos_sim) * self.APP_WEIGHT
        return cost_matrix

    def _match_cascade(self, detections):
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))

        distance_cost = self._get_distance_cost_matrix(detections)
        orientation_cost = self._get_orientation_cost_matrix(detections)
        appearance_cost = self._get_appearance_cost_matrix(detections)

        total_cost = distance_cost + orientation_cost + appearance_cost

        gate_mask = (distance_cost <= self.DISTANCE_THRESHOLD_PRIMARY).astype(np.int32)
        if gate_mask.sum() == 0:
            return [], track_indices, detection_indices

        gated_cost = total_cost.copy()
        gated_cost[gate_mask == 0] = 1e5

        row_indices, col_indices = hungarian_algorithm(gated_cost)

        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)

        for row, col in zip(row_indices, col_indices):
            if distance_cost[row, col] <= self.DISTANCE_THRESHOLD_PRIMARY:
                matches.append((row, col))
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)
                if col in unmatched_detections:
                    unmatched_detections.remove(col)

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, gt_positions=None, gt_ids=None):
        self.frame_count += 1
        self.predict()
        matches, unmatched_tracks, unmatched_detections = self._match_cascade(detections)

        for track_idx, det_idx in matches:
            det = detections[det_idx]
            if isinstance(det, Detection):
                x, y = det.center[0], det.center[1]
                feature = det.feature if np.any(det.feature) else None
                fusion_state = det.fusion_state
            else:
                x, y = det[0], det[1]
                feature = None
                fusion_state = FusionState.RADAR_ONLY
            self.tracks[track_idx].update(x, y, feature=feature, fusion_state=fusion_state)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            if isinstance(det, Detection):
                x, y = det.center[0], det.center[1]
                feature = det.feature if np.any(det.feature) else None
            else:
                x, y = det[0], det[1]
                feature = None
            self.tracks.append(KalmanTrack(x, y, feature=feature))

        self.tracks = [t for t in self.tracks if t.state != KalmanTrackState.Deleted]

    def get_confirmed_tracks(self):
        return [t for t in self.tracks if t.state == KalmanTrackState.Confirmed]
