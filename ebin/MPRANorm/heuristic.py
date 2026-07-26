import numpy as np


def get_normalized_counts(counts):
    K, num_bins = counts.shape
    M = counts.sum(axis=0)
    norm_factors = M / M.max()
    corrected_counts = counts / norm_factors[np.newaxis, :]
    return corrected_counts

def build_alpha_components(a, decay_rate, num_bins=4):
    alpha_components = list(); bin_indices = np.arange(num_bins)
    for i in range(num_bins):
        alpha_components.append(a * np.exp(-decay_rate * np.abs(bin_indices - i)))
    for i in range(num_bins - 1):
        alpha_components.append(a * np.exp(-decay_rate * np.abs(bin_indices - (i + 0.5))))
    return alpha_components

def fraction_estimator(counts, normalize: bool = False):
    if normalize:
        counts = get_normalized_counts(counts)
    row_sums = counts.sum(axis=1, keepdims=True)
    p_est = counts / (row_sums + 1e-9)
    return p_est

def fraction_map_estimator(counts, normalize: bool = False, a=10.0, decay_rate=0.5, significance_threshold=0.6):
    K, num_bins = counts.shape; epsilon = 1e-12
    if normalize:
        counts = get_normalized_counts(counts)
    alpha_components = build_alpha_components(a, decay_rate, num_bins)
    alpha_uniform = np.full(num_bins, a / num_bins)
    p_refined_est = np.zeros((K, num_bins))
    for k in range(K):
        c_k = counts[k, :]
        if c_k.sum() < 1e-6:
            p_refined_est[k, :] = np.full(num_bins, 1.0 / num_bins); continue
        sorted_indices = np.argsort(c_k)[::-1]; peak1_idx, peak2_idx = sorted_indices[0], sorted_indices[1]
        if abs(peak1_idx - peak2_idx) == 1 and c_k[peak2_idx] > c_k[peak1_idx] * significance_threshold:
            edge_idx = min(peak1_idx, peak2_idx); alpha_best = alpha_components[4 + edge_idx]
        elif abs(peak1_idx - peak2_idx) != 1 and c_k[peak2_idx] > 0.1 * c_k[peak1_idx]: 
             alpha_best = alpha_uniform
        else:
            alpha_best = alpha_components[peak1_idx]
        pseudo_counts = np.maximum(0, alpha_best - 1.0)
        numerator = c_k + pseudo_counts
        p_refined_est[k, :] = numerator / (np.sum(numerator) + epsilon)
    return p_refined_est

