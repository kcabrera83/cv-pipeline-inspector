"""Image feature extraction utilities (simulated for non-image input)."""

import numpy as np


class ImageFeatureExtractor:
    """Extracts simulated image features from raw pixel data or feature vectors."""

    FEATURE_NAMES = [
        "pixel_mean",
        "pixel_std",
        "edge_density",
        "texture_entropy",
        "color_r",
        "color_g",
        "color_b",
        "hog_energy",
        "laplacian_var",
        "gradient_magnitude",
        "color_intensity",
        "edge_texture_ratio",
        "color_variance",
        "contrast",
    ]

    def __init__(self):
        pass

    def extract_from_vector(self, feature_vector):
        """Ensure a feature vector has exactly 14 features."""
        arr = np.array(feature_vector, dtype=np.float64)
        if arr.shape[0] < 14:
            arr = np.pad(arr, (0, 14 - arr.shape[0]), mode="constant")
        elif arr.shape[0] > 14:
            arr = arr[:14]
        return arr

    def extract_from_intensity_distribution(self, pixel_values):
        """Compute features from a pixel intensity distribution array."""
        pixel_values = np.array(pixel_values, dtype=np.float64)
        pixel_mean = np.mean(pixel_values)
        pixel_std = np.std(pixel_values)

        hist, _ = np.histogram(pixel_values, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        nonzero = hist[hist > 0]
        texture_entropy = -np.sum(nonzero * np.log2(nonzero))

        return pixel_mean, pixel_std, texture_entropy

    def compute_edge_features(self, feature_vector):
        """Compute edge detection metrics from a simulated gradient map."""
        arr = np.array(feature_vector, dtype=np.float64)
        threshold = np.mean(arr) * 0.8
        edge_pixels = np.sum(arr > threshold)
        edge_density = edge_pixels / (len(arr) + 1e-10)
        gradient_magnitude = np.mean(np.abs(np.diff(arr)))
        laplacian_var = np.var(np.gradient(np.gradient(arr)))
        return edge_density, gradient_magnitude, laplacian_var

    def compute_color_histogram(self, r_values, g_values, b_values):
        """Compute color histogram features."""
        r_mean = np.mean(r_values)
        g_mean = np.mean(g_values)
        b_mean = np.mean(b_values)
        color_intensity = (r_mean + g_mean + b_mean) / 3.0
        color_variance = np.std([r_mean, g_mean, b_mean])
        return r_mean, g_mean, b_mean, color_intensity, color_variance

    def compute_hog_energy(self, feature_vector):
        """Compute simulated HOG energy from feature vector."""
        arr = np.array(feature_vector, dtype=np.float64)
        gradients = np.abs(np.diff(arr))
        hog_energy = np.sum(gradients ** 2) / (len(gradients) + 1e-10)
        hog_energy = min(1.0, hog_energy / (np.max(np.abs(arr)) + 1e-10))
        return hog_energy

    def analyze(self, feature_vector):
        """Full analysis pipeline returning all 14 features."""
        vec = self.extract_from_vector(feature_vector)
        pixel_mean, pixel_std, texture_entropy = self.extract_from_intensity_distribution(vec)
        edge_density, gradient_magnitude, laplacian_var = self.compute_edge_features(vec)
        hog_energy = self.compute_hog_energy(vec)
        r_mean, g_mean, b_mean, color_intensity, color_variance = self.compute_color_histogram(
            vec[:5] if len(vec) >= 5 else vec,
            vec[5:10] if len(vec) >= 10 else vec,
            vec[10:14] if len(vec) >= 14 else vec,
        )
        edge_texture_ratio = edge_density / (texture_entropy + 1e-6)
        contrast = pixel_std / (pixel_mean + 1e-6)

        return np.array([
            pixel_mean, pixel_std, edge_density, texture_entropy,
            r_mean, g_mean, b_mean, hog_energy,
            laplacian_var, gradient_magnitude,
            color_intensity, edge_texture_ratio, color_variance, contrast,
        ])


def histogram_analysis(pixel_values):
    """Analyze a pixel intensity histogram."""
    pixel_values = np.array(pixel_values, dtype=np.float64)
    hist, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-10)

    mean_intensity = np.mean(pixel_values)
    std_intensity = np.std(pixel_values)
    skewness = float(np.mean(((pixel_values - mean_intensity) / (std_intensity + 1e-10)) ** 3))
    kurtosis = float(np.mean(((pixel_values - mean_intensity) / (std_intensity + 1e-10)) ** 4) - 3)

    return {
        "mean": mean_intensity,
        "std": std_intensity,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "histogram": hist_norm.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def edge_detection_metrics(feature_vector):
    """Compute edge detection metrics from a feature vector."""
    arr = np.array(feature_vector, dtype=np.float64)
    threshold = np.mean(arr) * 0.7
    edges = (arr > threshold).astype(float)

    edge_density = np.mean(edges)
    gradient_magnitude = np.mean(np.abs(np.diff(arr)))
    laplacian_var = np.var(np.gradient(np.gradient(arr)))

    return {
        "edge_density": edge_density,
        "gradient_magnitude": gradient_magnitude,
        "laplacian_variance": laplacian_var,
    }
