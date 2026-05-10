from dataclasses import dataclass, replace
import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np

import pydeform.sitk_api as pydeform

from spinemira.core.filters import normalize_to_label_intensity_mode
from spinemira.core.io import load_image, load_label_map
from spinemira.core.morphology.straightening import straighten_coord
from spinemira.core.segmentation.utils import (
    filter_label_map,
    get_common_labels,
    get_level_coordinates,
    get_roi_mask,
)
from spinemira.io.bids import Layout, add_suffix_to_path_name


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageBundle:
    image: sitk.Image
    label_map: sitk.Image | None = None
    levels: sitk.Image | None = None
    mask: sitk.Image | None = None

    def with_mask(self, mask: sitk.Image) -> "ImageBundle":
        return replace(self, mask=mask)

    def with_label_map(self, label_map: sitk.Image) -> "ImageBundle":
        return replace(self, label_map=label_map)

    def with_levels(self, levels: sitk.Image) -> "ImageBundle":
        return replace(self, levels=levels)

    def straightened(
        self, *, si_padding_mm=30.0, target_curve_point_spacing_mm=30.0
    ) -> "ImageBundle":
        logger.info("Applying straightening.")

        if self.levels is None:
            raise ValueError("Levels need to be present to perform straightening.")

        level_coords = get_level_coordinates(self.levels)
        image, label_map, levels = straighten_coord(
            level_coords,
            [self.image, self.label_map, self.levels],
            orders=[1, 0, 0],
            si_padding_mm=si_padding_mm,
            target_curve_point_spacing_mm=target_curve_point_spacing_mm,
        )
        return replace(self, image=image, label_map=label_map, levels=levels)

    def filtered_to_common_labels(self, ref: "ImageBundle") -> "ImageBundle":
        logger.info("Applying filtering on common labels.")

        if self.label_map is not None and ref.label_map is not None:
            common_labels = get_common_labels(self.label_map, ref.label_map)
            filtered_label_map = filter_label_map(self.label_map, common_labels)

        if self.levels is not None and ref.levels is not None:
            common_levels = get_common_labels(self.levels, ref.levels)
            filtered_levels = filter_label_map(self.levels, common_levels)

        return replace(
            self,
            label_map=filtered_label_map,
            levels=filtered_levels,
        )

    def normalized_to_csf(self, csf_label: int) -> "ImageBundle":
        logger.info("Applying normalization to CSF.")

        if self.label_map is None:
            raise ValueError(
                "Label map needs to be present to perform normalization to CSF."
            )

        return replace(
            self,
            image=normalize_to_label_intensity_mode(
                self.image, self.label_map, csf_label
            ),
        )

    def with_roi_mask(
        self, *, dilate_radius: tuple[int, ...] = (8, 8, 8)
    ) -> "ImageBundle":
        logger.info("Applying mask based on convex region enclosing segmentation.")
        return self.with_mask(get_roi_mask(self.label_map, dilate_radius=dilate_radius))

    def with_harmonized_directions(self, *, tol: float = 1e-3) -> "ImageBundle":
        logger.info("Harmonizing directions.")
        ref_direction = np.array(self.image.GetDirection())

        def harmonize(image: sitk.Image):
            if image is None:
                return None

            current_direction = np.array(image.GetDirection())
            if not np.allclose(current_direction, ref_direction, atol=tol):
                max_diff = np.max(np.abs(current_direction - ref_direction))
                raise ValueError(
                    f"Direction mismatch, image direction exceeds tolerance {tol:.1e} "
                    f"(max difference: {max_diff:.2e}). Cannot harmonize directions automatically."
                )
            image.SetDirection(tuple(ref_direction))

            return image

        return replace(
            self,
            label_map=None if self.label_map is None else harmonize(self.label_map),
            levels=None if self.levels is None else harmonize(self.levels),
            mask=None if self.mask is None else harmonize(self.mask),
        )

    def apply_deformation_field(self, df: sitk.Image) -> "ImageBundle":
        logger.info("Applying deformation field.")
        return replace(
            self,
            image=pydeform.transform(self.image, df),
            label_map=pydeform.transform(
                self.label_map, df, interp=sitk.sitkNearestNeighbor
            ),
            levels=pydeform.transform(self.levels, df, interp=sitk.sitkNearestNeighbor),
            mask=pydeform.transform(self.mask, df, interp=sitk.sitkNearestNeighbor),
        )

    def apply_transform(self, ref: sitk.Image, tf: sitk.Transform) -> "ImageBundle":
        logger.info("Applying transform.")
        return replace(
            self,
            image=sitk.Resample(self.image, ref, tf),
            label_map=None
            if self.label_map is None
            else sitk.Resample(self.label_map, ref, tf, sitk.sitkNearestNeighbor),
            levels=None
            if self.levels is None
            else sitk.Resample(self.levels, ref, tf, sitk.sitkNearestNeighbor),
            mask=None
            if self.mask is None
            else sitk.Resample(self.mask, ref, tf, sitk.sitkNearestNeighbor),
        )

    def save_to_same_directory(
        self,
        image_path: Path,
        label_map_suffix: str = "dseg",
        levels_suffix: str = "levels",
        mask_suffix: str = "mask",
    ):
        image_path.parent.mkdir(exist_ok=True, parents=True)

        label_map_path = add_suffix_to_path_name(image_path, label_map_suffix)
        levels_path = add_suffix_to_path_name(image_path, levels_suffix)
        mask_path = add_suffix_to_path_name(image_path, mask_suffix)

        sitk.WriteImage(self.image, image_path)

        if self.label_map:
            sitk.WriteImage(self.label_map, label_map_path)

        if self.levels:
            sitk.WriteImage(self.levels, levels_path)

        if self.mask:
            sitk.WriteImage(self.mask, mask_path)

    @staticmethod
    def load_from_same_directory(
        image_path: Path,
        label_map_suffix: str = "dseg",
        levels_suffix: str = "levels",
        mask_suffix: str = "mask",
    ) -> "ImageBundle":
        label_map_path = add_suffix_to_path_name(image_path, label_map_suffix)
        levels_path = add_suffix_to_path_name(image_path, levels_suffix)
        mask_path = add_suffix_to_path_name(image_path, mask_suffix)

        return ImageBundle(
            image=load_image(image_path),
            label_map=load_label_map(label_map_path),
            levels=load_label_map(levels_path),
            mask=load_label_map(mask_path),
        )


@dataclass
class PipelineConfig:
    dataset_root: Path
    output_derivative_name: str
    image_weight: str
    load_sidecars: bool
    overwrite_existing: bool


def build_image_query(base: str, image_weight: str) -> str:
    extra = [
        "~is_sidecar",
        f"`suffix` == '{image_weight}'",
    ]
    return " and ".join([base, *extra])


def index_dataset(cnf: PipelineConfig) -> Layout:
    logging.info("Loading dataset...")
    layout = Layout(root=cnf.dataset_root, include_derivatives=True)
    layout.index(load_sidecars=cnf.load_sidecars)
    logging.info("Dataset indexed.")

    return layout
