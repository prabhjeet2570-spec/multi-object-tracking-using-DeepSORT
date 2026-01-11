import numpy as np
from kalman import MotionPredictor
from matching import hierarchical_matching, solve_assignment_problem, apply_measurement_gate, compute_iou_cost


class TrackStatus:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class ObjectTrack:
    def __init__(self, initial_mean, initial_covariance, unique_id, confirmation_threshold, 
                 deletion_threshold, initial_feature=None, object_category=None):
        self.state_mean = initial_mean
        self.state_covariance = initial_covariance
        self.unique_id = unique_id
        self.total_hits = 1
        self.track_age = 1
        self.frames_since_update = 0

        self.status = TrackStatus.Tentative
        self.appearance_history = []
        if initial_feature is not None:
            self.appearance_history.append(initial_feature)

        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold

        self.original_category = object_category
        self.object_category = object_category

    def to_top_left_width_height(self):
        box_representation = self.state_mean[:4].copy()
        box_representation[2] *= box_representation[3]
        box_representation[:2] -= box_representation[2:] / 2
        return box_representation

    def to_top_left_bottom_right(self):
        box_tlwh = self.to_top_left_width_height()
        box_tlbr = box_tlwh.copy()
        box_tlbr[2:] = box_tlbr[:2] + box_tlbr[2:]
        return box_tlbr

    def get_object_category(self):
        return self.object_category

    def predict_next_position(self, motion_model):
        self.state_mean, self.state_covariance = motion_model.predict_next_state(
            self.state_mean, self.state_covariance)
        self.track_age += 1
        self.frames_since_update += 1

    def update_with_detection(self, motion_model, new_detection):
        self.state_mean, self.state_covariance = motion_model.update_with_measurement(
            self.state_mean, self.state_covariance, new_detection.to_center_aspect_height())
        self.appearance_history.append(new_detection.appearance_feature)

        self.total_hits += 1
        self.frames_since_update = 0

        if self.status == TrackStatus.Tentative and self.total_hits >= self.confirmation_threshold:
            self.status = TrackStatus.Confirmed

    def mark_as_missed(self):
        if self.status == TrackStatus.Tentative:
            self.status = TrackStatus.Deleted
        elif self.frames_since_update > self.deletion_threshold:
            self.status = TrackStatus.Deleted

    def is_tentative(self):
        return self.status == TrackStatus.Tentative

    def is_confirmed(self):
        return self.status == TrackStatus.Confirmed

    def is_deleted(self):
        return self.status == TrackStatus.Deleted


class MultiObjectTracker:
    def __init__(self, appearance_metric, iou_threshold=0.7, max_track_age=30, min_hits=3):
        self.appearance_metric = appearance_metric
        self.iou_threshold = iou_threshold
        self.max_track_age = max_track_age
        self.min_hits = min_hits

        self.motion_model = MotionPredictor()
        self.active_tracks = []
        self.next_track_id = 1

    def predict_all_tracks(self):
        for single_track in self.active_tracks:
            single_track.predict_next_position(self.motion_model)

    def update_with_detections(self, new_detections):
        matched_pairs, unmatched_tracks, unmatched_detections = self.associate_detections(new_detections)

        for track_idx, detection_idx in matched_pairs:
            self.active_tracks[track_idx].update_with_detection(
                self.motion_model, new_detections[detection_idx])
        
        for track_idx in unmatched_tracks:
            self.active_tracks[track_idx].mark_as_missed()
        
        for detection_idx in unmatched_detections:
            self.create_new_track(new_detections[detection_idx])
        
        self.active_tracks = [track for track in self.active_tracks if not track.is_deleted()]

        confirmed_ids = [track.unique_id for track in self.active_tracks if track.is_confirmed()]
        all_features = []
        all_labels = []
        
        for single_track in self.active_tracks:
            if not single_track.is_confirmed():
                continue
            all_features += single_track.appearance_history
            all_labels += [single_track.unique_id for _ in single_track.appearance_history]
            single_track.appearance_history = []
        
        self.appearance_metric.update_gallery(
            np.asarray(all_features), np.asarray(all_labels), confirmed_ids)

    def associate_detections(self, new_detections):
        def compute_appearance_cost(tracks, detections, track_indices, detection_indices):
            detection_features = np.array([detections[idx].appearance_feature for idx in detection_indices])
            track_ids = np.array([tracks[idx].unique_id for idx in track_indices])
            cost_matrix = self.appearance_metric.compute_distance_matrix(detection_features, track_ids)
            cost_matrix = apply_measurement_gate(
                self.motion_model, cost_matrix, tracks, detections, track_indices, detection_indices)
            return cost_matrix

        confirmed_indices = [idx for idx, track in enumerate(self.active_tracks) if track.is_confirmed()]
        unconfirmed_indices = [idx for idx, track in enumerate(self.active_tracks) if not track.is_confirmed()]

        primary_matches, unmatched_from_primary, remaining_detections = hierarchical_matching(
            compute_appearance_cost, self.appearance_metric.threshold_value, self.max_track_age,
            self.active_tracks, new_detections, confirmed_indices)

        iou_candidates = unconfirmed_indices + [
            idx for idx in unmatched_from_primary
            if self.active_tracks[idx].frames_since_update == 1]
        
        still_unmatched_tracks = [
            idx for idx in unmatched_from_primary
            if self.active_tracks[idx].frames_since_update != 1]
        
        secondary_matches, unmatched_from_secondary, remaining_detections = solve_assignment_problem(
            compute_iou_cost, self.iou_threshold, self.active_tracks,
            new_detections, iou_candidates, remaining_detections)

        all_matches = primary_matches + secondary_matches
        all_unmatched_tracks = list(set(still_unmatched_tracks + unmatched_from_secondary))
        
        return all_matches, all_unmatched_tracks, remaining_detections

    def create_new_track(self, detection):
        initial_mean, initial_covariance = self.motion_model.initialize_track(
            detection.to_center_aspect_height())
        detection_category = detection.get_class_name()
        
        new_track = ObjectTrack(
            initial_mean, initial_covariance, self.next_track_id, self.min_hits, 
            self.max_track_age, detection.appearance_feature, detection_category)
        
        self.active_tracks.append(new_track)
        self.next_track_id += 1
