import logging
import SimpleITK as sitk
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def calc_delta_mu(
    image: sitk.Image,
    label_map: sitk.Image,
    label: int | None = None,
    n_replicates: int = 100,
    max_iter: int = 2000,
) -> float:
    """
    Calculate the delta mu metric for intervertebral disc degeneration classification.

    The delta mu metric is the absolute difference between the means of two Gaussian
    distributions fitted to the intensity values of a segmented region. This metric
    is based on the methodology described in [1]_.

    Parameters
    ----------
    image : sitk.Image
        The input image containing intensity values. The masked image should ideally
        represent a distribution that can be modeled by two Gaussian distributions.
    label_map : sitk.Image
        The label map image where segmented structures are labeled.
    label : int | None, optional
        Label to use for masking. If not specified, non-zero values of the input
        label map are used.
    n_replicates : int, optional
        Number of averages, by default 100.
    max_iter : int, optional
        Max iterations, by default 2000.

    Returns
    -------
    float
        Difference between the means of the two fitted Gaussian distributions.

    References
    ----------
    .. [1] Waldenberg, Christian & Hebelka, Hanna & Brisby, Helena & Lagerstrand, Kerstin.
        (2018). MRI histogram analysis enables objective and continuous classification of
        intervertebral disc degeneration. European Spine Journal. 27.
        10.1007/s00586-017-5264-7.
    """

    image_arr = sitk.GetArrayViewFromImage(image)
    label_map_arr = sitk.GetArrayViewFromImage(label_map)

    if label is None:
        mask = label_map_arr != 0
    else:
        mask = label_map_arr == label

    data = image_arr[mask].reshape(-1, 1)

    try:
        gm = GaussianMixture(
            n_components=2,
            n_init=n_replicates,
            max_iter=max_iter,
            init_params="k-means++",
        )
        gm.fit(data)
    except ConvergenceWarning:
        logger.warning(
            "There was an error fitting the Gaussian mixture model. Retrying with regularization set to 0.01"
        )
        gm = GaussianMixture(
            n_components=2,
            n_init=n_replicates,
            max_iter=max_iter,
            init_params="k-means++",
            reg_covar=0.01,
        )

    gm.get_params

    mu = gm.means_.flatten()

    return abs(mu[0] - mu[1])
