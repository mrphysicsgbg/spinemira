from unittest import TestCase
import SimpleITK as sitk
import numpy as np
from scipy.stats import wasserstein_distance

from spinemira.core.filters import histogram_matching


class TestHistogramMatchingFilter(TestCase):
    @staticmethod
    def _wasserstein_histogram_distance(
        image_a: sitk.Image, image_b: sitk.Image
    ) -> float:
        array_a = sitk.GetArrayViewFromImage(image_a).flatten()
        array_b = sitk.GetArrayViewFromImage(image_b).flatten()

        value_min = np.concat((array_a, array_b)).min()
        value_max = np.concat((array_a, array_b)).max()

        hist_a, edges = np.histogram(array_a, range=(value_min, value_max), bins=64)
        hist_b, edges = np.histogram(array_b, range=(value_min, value_max), bins=64)

        prob_weight_a = hist_a.astype(np.float64) / hist_a.sum()
        prob_weight_b = hist_b.astype(np.float64) / hist_b.sum()

        centers = 0.5 * (edges[:-1] + edges[1:])

        return wasserstein_distance(
            centers, centers, u_weights=prob_weight_a, v_weights=prob_weight_b
        )

    def setUp(self):
        """Setup synthetic images."""
        rng = np.random.RandomState(10)
        self.source_image = sitk.GetImageFromArray(
            rng.normal(loc=5, scale=2, size=(64, 64)).astype(np.float32)
        )
        self.reference_image = sitk.GetImageFromArray(
            np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
        )

    def test_histogram_matching(self):
        """Test if the Wasserstein distance of the histograms has decreased after histogram matching."""
        filtered_image = histogram_matching(self.source_image, self.reference_image)

        distance_source = self._wasserstein_histogram_distance(
            self.source_image, self.reference_image
        )
        distance_filtered = self._wasserstein_histogram_distance(
            filtered_image, self.reference_image
        )

        self.assertLess(distance_filtered, distance_source)
