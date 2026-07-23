"""Synthetic image feature generator for pipeline defect simulation."""

import numpy as np


DEFECT_TYPES = ["corrosion", "cracks", "dents", "leaks", "weld_defects", "healthy"]

DEFECT_PROFILES = {
    "corrosion": {
        "pixel_mean": (90, 140),
        "pixel_std": (25, 50),
        "edge_density": (0.15, 0.40),
        "texture_entropy": (3.5, 6.0),
        "color_r": (120, 200),
        "color_g": (60, 120),
        "color_b": (30, 80),
        "hog_energy": (0.3, 0.7),
        "laplacian_var": (800, 2500),
        "gradient_magnitude": (15, 45),
    },
    "cracks": {
        "pixel_mean": (60, 110),
        "pixel_std": (30, 55),
        "edge_density": (0.35, 0.65),
        "texture_entropy": (4.0, 6.5),
        "color_r": (40, 90),
        "color_g": (40, 90),
        "color_b": (40, 90),
        "hog_energy": (0.5, 0.9),
        "laplacian_var": (1500, 4000),
        "gradient_magnitude": (25, 60),
    },
    "dents": {
        "pixel_mean": (100, 150),
        "pixel_std": (15, 35),
        "edge_density": (0.10, 0.25),
        "texture_entropy": (2.5, 4.5),
        "color_r": (110, 160),
        "color_g": (110, 160),
        "color_b": (100, 150),
        "hog_energy": (0.2, 0.5),
        "laplacian_var": (400, 1200),
        "gradient_magnitude": (10, 30),
    },
    "leaks": {
        "pixel_mean": (40, 100),
        "pixel_std": (20, 45),
        "edge_density": (0.20, 0.50),
        "texture_entropy": (3.0, 5.5),
        "color_r": (30, 80),
        "color_g": (50, 110),
        "color_b": (60, 130),
        "hog_energy": (0.4, 0.8),
        "laplacian_var": (600, 2000),
        "gradient_magnitude": (18, 50),
    },
    "weld_defects": {
        "pixel_mean": (130, 190),
        "pixel_std": (20, 40),
        "edge_density": (0.25, 0.55),
        "texture_entropy": (3.5, 5.8),
        "color_r": (150, 220),
        "color_g": (130, 180),
        "color_b": (100, 150),
        "hog_energy": (0.45, 0.85),
        "laplacian_var": (1000, 3000),
        "gradient_magnitude": (20, 55),
    },
    "healthy": {
        "pixel_mean": (140, 180),
        "pixel_std": (10, 25),
        "edge_density": (0.05, 0.15),
        "texture_entropy": (1.5, 3.5),
        "color_r": (140, 180),
        "color_g": (140, 180),
        "color_b": (130, 170),
        "hog_energy": (0.1, 0.3),
        "laplacian_var": (100, 500),
        "gradient_magnitude": (5, 18),
    },
}

SEVERITY_PROFILES = {
    "corrosion": (3, 9),
    "cracks": (4, 10),
    "dents": (2, 7),
    "leaks": (5, 10),
    "weld_defects": (3, 9),
    "healthy": (0, 1),
}


def _uniform_range(low, high, size, rng):
    return rng.uniform(low, high, size)


def generate_samples(n_per_class=200, seed=42):
    """Generate synthetic feature vectors simulating pipeline image analysis."""
    rng = np.random.RandomState(seed)
    all_features = []
    all_labels = []
    all_severities = []

    for defect_type in DEFECT_TYPES:
        profile = DEFECT_PROFILES[defect_type]
        sev_lo, sev_hi = SEVERITY_PROFILES[defect_type]

        for _ in range(n_per_class):
            pixel_mean = _uniform_range(*profile["pixel_mean"], 1, rng)[0]
            pixel_std = _uniform_range(*profile["pixel_std"], 1, rng)[0]
            edge_density = _uniform_range(*profile["edge_density"], 1, rng)[0]
            texture_entropy = _uniform_range(*profile["texture_entropy"], 1, rng)[0]
            color_r = _uniform_range(*profile["color_r"], 1, rng)[0]
            color_g = _uniform_range(*profile["color_g"], 1, rng)[0]
            color_b = _uniform_range(*profile["color_b"], 1, rng)[0]
            hog_energy = _uniform_range(*profile["hog_energy"], 1, rng)[0]
            laplacian_var = _uniform_range(*profile["laplacian_var"], 1, rng)[0]
            gradient_magnitude = _uniform_range(*profile["gradient_magnitude"], 1, rng)[0]

            color_intensity = (color_r + color_g + color_b) / 3.0
            edge_texture_ratio = edge_density / (texture_entropy + 1e-6)
            color_variance = np.std([color_r, color_g, color_b])
            contrast = pixel_std / (pixel_mean + 1e-6)

            features = [
                pixel_mean,
                pixel_std,
                edge_density,
                texture_entropy,
                color_r,
                color_g,
                color_b,
                hog_energy,
                laplacian_var,
                gradient_magnitude,
                color_intensity,
                edge_texture_ratio,
                color_variance,
                contrast,
            ]

            severity = round(rng.uniform(sev_lo, sev_hi), 1)
            severity = max(0, min(10, severity))

            all_features.append(features)
            all_labels.append(defect_type)
            all_severities.append(severity)

    return np.array(all_features), np.array(all_labels), np.array(all_severities)


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
