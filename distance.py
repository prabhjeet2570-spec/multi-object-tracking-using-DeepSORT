import numpy as np


def compute_pairwise_distances(features_a, features_b):
    features_a = np.asarray(features_a)
    features_b = np.asarray(features_b)
    
    if len(features_a) == 0 or len(features_b) == 0:
        return np.zeros((len(features_a), len(features_b)))
    
    squared_a = np.square(features_a).sum(axis=1)
    squared_b = np.square(features_b).sum(axis=1)
    squared_distances = -2. * np.dot(features_a, features_b.T) + squared_a[:, None] + squared_b[None, :]
    squared_distances = np.clip(squared_distances, 0., float(np.inf))
    
    return squared_distances


def compute_angular_distance(features_a, features_b, already_normalized=False):
    if not already_normalized:
        features_a = np.asarray(features_a) / np.linalg.norm(features_a, axis=1, keepdims=True)
        features_b = np.asarray(features_b) / np.linalg.norm(features_b, axis=1, keepdims=True)
    return 1. - np.dot(features_a, features_b.T)


def nearest_neighbor_euclidean(gallery_features, query_features):
    pairwise_distances = compute_pairwise_distances(gallery_features, query_features)
    return np.maximum(0.0, pairwise_distances.min(axis=0))


def nearest_neighbor_cosine(gallery_features, query_features):
    angular_distances = compute_angular_distance(gallery_features, query_features)
    return angular_distances.min(axis=0)


class AppearanceMetric:
    def __init__(self, distance_type, threshold_value, memory_size=None):
        if distance_type == "euclidean":
            self.distance_function = nearest_neighbor_euclidean
        elif distance_type == "cosine":
            self.distance_function = nearest_neighbor_cosine
        else:
            raise ValueError("Distance type must be either 'euclidean' or 'cosine'")
        
        self.threshold_value = threshold_value
        self.memory_size = memory_size
        self.feature_gallery = {}

    def update_gallery(self, new_features, identity_labels, active_identities):
        for single_feature, identity_label in zip(new_features, identity_labels):
            self.feature_gallery.setdefault(identity_label, []).append(single_feature)
            if self.memory_size is not None:
                self.feature_gallery[identity_label] = self.feature_gallery[identity_label][-self.memory_size:]
        
        self.feature_gallery = {identity: self.feature_gallery[identity] for identity in active_identities}

    def compute_distance_matrix(self, query_features, identity_labels):
        distance_matrix = np.zeros((len(identity_labels), len(query_features)))
        for row_idx, identity_label in enumerate(identity_labels):
            distance_matrix[row_idx, :] = self.distance_function(
                self.feature_gallery[identity_label], query_features)
        return distance_matrix
