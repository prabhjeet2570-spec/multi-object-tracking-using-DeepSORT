import numpy as np
import scipy.linalg


class MotionPredictor:
    def __init__(self):
        state_dimensions = 4
        time_step = 1.
        
        self.transition_matrix = np.eye(2 * state_dimensions, 2 * state_dimensions)
        for dimension in range(state_dimensions):
            self.transition_matrix[dimension, state_dimensions + dimension] = time_step
        self.observation_matrix = np.eye(state_dimensions, 2 * state_dimensions)
        self.position_noise_weight = 1. / 20
        self.velocity_noise_weight = 1. / 160

    def initialize_track(self, initial_measurement):
        initial_position = initial_measurement
        initial_velocity = np.zeros_like(initial_position)
        state_mean = np.r_[initial_position, initial_velocity]

        position_uncertainty = [
            2 * self.position_noise_weight * initial_measurement[3],
            2 * self.position_noise_weight * initial_measurement[3],
            1e-2,
            2 * self.position_noise_weight * initial_measurement[3],
            10 * self.velocity_noise_weight * initial_measurement[3],
            10 * self.velocity_noise_weight * initial_measurement[3],
            1e-5,
            10 * self.velocity_noise_weight * initial_measurement[3]]
        state_covariance = np.diag(np.square(position_uncertainty))
        return state_mean, state_covariance

    def predict_next_state(self, current_mean, current_covariance):
        position_noise = [
            self.position_noise_weight * current_mean[3],
            self.position_noise_weight * current_mean[3],
            1e-2,
            self.position_noise_weight * current_mean[3]]
        velocity_noise = [
            self.velocity_noise_weight * current_mean[3],
            self.velocity_noise_weight * current_mean[3],
            1e-5,
            self.velocity_noise_weight * current_mean[3]]
        process_noise = np.diag(np.square(np.r_[position_noise, velocity_noise]))

        predicted_mean = np.dot(self.transition_matrix, current_mean)
        predicted_covariance = np.linalg.multi_dot((
            self.transition_matrix, current_covariance, self.transition_matrix.T)) + process_noise

        return predicted_mean, predicted_covariance

    def project_to_measurement_space(self, state_mean, state_covariance):
        measurement_noise = [
            self.position_noise_weight * state_mean[3],
            self.position_noise_weight * state_mean[3],
            1e-1,
            self.position_noise_weight * state_mean[3]]
        measurement_covariance = np.diag(np.square(measurement_noise))

        projected_mean = np.dot(self.observation_matrix, state_mean)
        projected_covariance = np.linalg.multi_dot((
            self.observation_matrix, state_covariance, self.observation_matrix.T))
        return projected_mean, projected_covariance + measurement_covariance

    def update_with_measurement(self, prior_mean, prior_covariance, new_measurement):
        projected_mean, projected_covariance = self.project_to_measurement_space(prior_mean, prior_covariance)

        cholesky_factor, is_lower = scipy.linalg.cho_factor(
            projected_covariance, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (cholesky_factor, is_lower), np.dot(prior_covariance, self.observation_matrix.T).T,
            check_finite=False).T
        measurement_residual = new_measurement - projected_mean

        posterior_mean = prior_mean + np.dot(measurement_residual, kalman_gain.T)
        posterior_covariance = prior_covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))
        return posterior_mean, posterior_covariance

    def compute_gating_distance(self, state_mean, state_covariance, measurements, position_only=False):
        projected_mean, projected_covariance = self.project_to_measurement_space(state_mean, state_covariance)
        if position_only:
            projected_mean, projected_covariance = projected_mean[:2], projected_covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_decomposition = np.linalg.cholesky(projected_covariance)
        measurement_differences = measurements - projected_mean
        normalized_differences = scipy.linalg.solve_triangular(
            cholesky_decomposition, measurement_differences.T, lower=True, check_finite=False,
            overwrite_b=True)
        mahalanobis_squared = np.sum(normalized_differences * normalized_differences, axis=0)
        return mahalanobis_squared
