import numpy as np
from scipy.optimize import linear_sum_assignment
from settings import CHI_SQUARE_95


INFINITY_COST = 1e+5


def solve_assignment_problem(cost_function, distance_limit, active_tracks, new_detections, 
                            track_subset=None, detection_subset=None):
    if track_subset is None:
        track_subset = np.arange(len(active_tracks))
    if detection_subset is None:
        detection_subset = np.arange(len(new_detections))

    if len(detection_subset) == 0 or len(track_subset) == 0:
        return [], track_subset, detection_subset

    cost_matrix = cost_function(active_tracks, new_detections, track_subset, detection_subset)
    cost_matrix[cost_matrix > distance_limit] = distance_limit + 1e-5
    assignment_indices = linear_sum_assignment(cost_matrix)
    assignment_indices = np.asarray(assignment_indices)
    assignment_indices = np.transpose(assignment_indices)
    
    matched_pairs = []
    unmatched_tracks = []
    unmatched_detections = []
    
    for column_idx, detection_id in enumerate(detection_subset):
        if column_idx not in assignment_indices[:, 1]:
            unmatched_detections.append(detection_id)
    
    for row_idx, track_id in enumerate(track_subset):
        if row_idx not in assignment_indices[:, 0]:
            unmatched_tracks.append(track_id)
    
    for row_idx, column_idx in assignment_indices:
        track_id = track_subset[row_idx]
        detection_id = detection_subset[column_idx]
        if cost_matrix[row_idx, column_idx] > distance_limit:
            unmatched_tracks.append(track_id)
            unmatched_detections.append(detection_id)
        else:
            matched_pairs.append((track_id, detection_id))
    
    return matched_pairs, unmatched_tracks, unmatched_detections


def hierarchical_matching(cost_function, distance_limit, max_depth, active_tracks, 
                         new_detections, track_subset=None, detection_subset=None):
    if track_subset is None:
        track_subset = list(range(len(active_tracks)))
    if detection_subset is None:
        detection_subset = list(range(len(new_detections)))

    remaining_detections = detection_subset
    all_matches = []
    
    for current_level in range(max_depth):
        if len(remaining_detections) == 0:
            break

        tracks_at_level = [
            track_idx for track_idx in track_subset
            if active_tracks[track_idx].frames_since_update == 1 + current_level
        ]
        
        if len(tracks_at_level) == 0:
            continue

        level_matches, _, remaining_detections = solve_assignment_problem(
            cost_function, distance_limit, active_tracks, new_detections,
            tracks_at_level, remaining_detections)
        all_matches += level_matches
    
    unmatched_tracks = list(set(track_subset) - set(track_id for track_id, _ in all_matches))
    return all_matches, unmatched_tracks, remaining_detections


def apply_measurement_gate(motion_predictor, cost_matrix, active_tracks, new_detections, 
                          track_subset, detection_subset, gate_cost=INFINITY_COST, position_only=False):
    gate_dimensions = 2 if position_only else 4
    gate_threshold = CHI_SQUARE_95[gate_dimensions]
    
    measurement_vectors = np.asarray(
        [new_detections[det_idx].to_center_aspect_height() for det_idx in detection_subset])
    
    for matrix_row, track_idx in enumerate(track_subset):
        current_track = active_tracks[track_idx]
        distances = motion_predictor.compute_gating_distance(
            current_track.state_mean, current_track.state_covariance, 
            measurement_vectors, position_only)
        cost_matrix[matrix_row, distances > gate_threshold] = gate_cost
    
    return cost_matrix


def calculate_intersection_over_union(target_box, candidate_boxes):
    target_top_left = target_box[:2]
    target_bottom_right = target_box[:2] + target_box[2:]
    candidates_top_left = candidate_boxes[:, :2]
    candidates_bottom_right = candidate_boxes[:, :2] + candidate_boxes[:, 2:]

    intersection_top_left = np.c_[
        np.maximum(target_top_left[0], candidates_top_left[:, 0])[:, np.newaxis],
        np.maximum(target_top_left[1], candidates_top_left[:, 1])[:, np.newaxis]]
    intersection_bottom_right = np.c_[
        np.minimum(target_bottom_right[0], candidates_bottom_right[:, 0])[:, np.newaxis],
        np.minimum(target_bottom_right[1], candidates_bottom_right[:, 1])[:, np.newaxis]]
    intersection_dimensions = np.maximum(0., intersection_bottom_right - intersection_top_left)

    intersection_area = intersection_dimensions.prod(axis=1)
    target_area = target_box[2:].prod()
    candidate_areas = candidate_boxes[:, 2:].prod(axis=1)
    union_area = target_area + candidate_areas - intersection_area
    
    return intersection_area / union_area


def compute_iou_cost(active_tracks, new_detections, track_subset=None, detection_subset=None):
    if track_subset is None:
        track_subset = np.arange(len(active_tracks))
    if detection_subset is None:
        detection_subset = np.arange(len(new_detections))

    cost_matrix = np.zeros((len(track_subset), len(detection_subset)))
    
    for matrix_row, track_idx in enumerate(track_subset):
        if active_tracks[track_idx].frames_since_update > 1:
            cost_matrix[matrix_row, :] = INFINITY_COST
            continue

        track_box = active_tracks[track_idx].to_top_left_width_height()
        detection_boxes = np.asarray([new_detections[det_idx].bounding_box for det_idx in detection_subset])
        cost_matrix[matrix_row, :] = 1. - calculate_intersection_over_union(track_box, detection_boxes)
    
    return cost_matrix


def suppress_overlapping_detections(detection_boxes, detection_classes, overlap_threshold, confidence_scores=None):
    if len(detection_boxes) == 0:
        return []

    boxes = detection_boxes.astype(np.float64)
    selected_indices = []

    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2] + boxes[:, 0]
    y_max = boxes[:, 3] + boxes[:, 1]

    box_areas = (x_max - x_min + 1) * (y_max - y_min + 1)
    
    if confidence_scores is not None:
        sorted_indices = np.argsort(confidence_scores)
    else:
        sorted_indices = np.argsort(y_max)

    while len(sorted_indices) > 0:
        last_position = len(sorted_indices) - 1
        current_index = sorted_indices[last_position]
        selected_indices.append(current_index)

        overlap_x_min = np.maximum(x_min[current_index], x_min[sorted_indices[:last_position]])
        overlap_y_min = np.maximum(y_min[current_index], y_min[sorted_indices[:last_position]])
        overlap_x_max = np.minimum(x_max[current_index], x_max[sorted_indices[:last_position]])
        overlap_y_max = np.minimum(y_max[current_index], y_max[sorted_indices[:last_position]])

        overlap_width = np.maximum(0, overlap_x_max - overlap_x_min + 1)
        overlap_height = np.maximum(0, overlap_y_max - overlap_y_min + 1)

        overlap_ratio = (overlap_width * overlap_height) / box_areas[sorted_indices[:last_position]]

        sorted_indices = np.delete(
            sorted_indices, np.concatenate(
                ([last_position], np.where(overlap_ratio > overlap_threshold)[0])))

    return selected_indices
