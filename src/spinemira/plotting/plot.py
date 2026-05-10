import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
from IPython.display import display


def plot_sitk_images_with_slider(
    images,
    titles=None,
    cmaps=None,
    suptitle=None,
    axis=2,
    figsize=(10, 5),
    heatmaps=None,
    heatmaps_cmaps=None,
    thresholds=None,
    norms=None,
    heatmap_alpha=0.5,
    show_colorbars=True,
    num_cols=None,
):
    if not images:
        raise ValueError("No images provided.")

    def to_numpy(img):
        return sitk.GetArrayFromImage(img) if isinstance(img, sitk.Image) else img

    np_images = [to_numpy(img) for img in images]
    image_shape = np_images[0].shape
    max_index = image_shape[axis]

    np_heatmaps = [to_numpy(hm) for hm in heatmaps] if heatmaps else None
    if np_heatmaps and len(np_heatmaps) != len(np_images):
        raise ValueError("Number of heatmaps must match number of images.")

    thresholds = thresholds or [None] * len(np_images)
    if len(thresholds) != len(np_images):
        raise ValueError("Number of thresholds must match number of images.")

    norms = norms or [None] * len(np_images)
    if len(norms) != len(np_images):
        raise ValueError("Number of norms must match number of images.")

    vmins = [np.nanmin(img) for img in np_images]
    vmaxs = [np.nanmax(img) for img in np_images]

    if np_heatmaps:
        heatmap_mins = [np.nanmin(hm) for hm in np_heatmaps]
        heatmap_maxs = [np.nanmax(hm) for hm in np_heatmaps]

    cols = num_cols if num_cols is not None else 2
    rows = -(-len(np_images) // cols)

    # Set up figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    axs = list(np.array(axs).flatten())

    fig.tight_layout()
    if suptitle:
        fig.suptitle(suptitle)

    def get_slice(volume, idx):
        slicer = [slice(None)] * 3
        slicer[axis] = idx
        return volume[tuple(slicer)].copy()

    def add_colorbar(ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    ims, heatmap_ims = [], []

    for ax in axs:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    for i, (img, vmin, vmax) in enumerate(zip(np_images, vmins, vmaxs)):
        ax = axs[i]
        ax.set_title(titles[i] if titles and i < len(titles) else f"Image {i + 1}")
        ax.set_aspect("equal")

        slice_2d = get_slice(img, 0)
        cmap = cmaps[i] if cmaps and i < len(cmaps) else "gray"
        im = ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax)
        ims.append(im)

        # Heatmap overlay if present
        if np_heatmaps:
            heatmap = np_heatmaps[i]
            threshold = thresholds[i]
            hm_slice = get_slice(heatmap, 0)

            if threshold is not None:
                hm_slice[hm_slice < threshold] = np.NaN

            hm_cmap = (
                heatmaps_cmaps[i]
                if heatmaps_cmaps and i < len(heatmaps_cmaps)
                else "jet"
            )
            hm_im = ax.imshow(
                hm_slice,
                cmap=hm_cmap,
                alpha=heatmap_alpha,
                vmin=heatmap_mins[i] if norms[i] is None else None,
                vmax=heatmap_maxs[i] if norms[i] is None else None,
                norm=norms[i],
                interpolation=None,
            )
            heatmap_ims.append(hm_im)
            if show_colorbars:
                add_colorbar(ax, hm_im)
        else:
            heatmap_ims.append(None)
            if show_colorbars:
                add_colorbar(ax, im)

        ax.invert_yaxis()

    # Slider widget
    slider = widgets.IntSlider(value=0, min=0, max=max_index - 1, description="Slice")

    def update_slice(index):
        for i, img in enumerate(np_images):
            ims[i].set_data(get_slice(img, index))
            if np_heatmaps and heatmap_ims[i] is not None:
                hm_slice = get_slice(np_heatmaps[i], index)
                threshold = thresholds[i]

                if threshold is not None:
                    hm_slice[hm_slice < threshold] = np.NaN

                heatmap_ims[i].set_data(hm_slice)
        fig.canvas.draw_idle()

    slider.observe(lambda change: update_slice(change["new"]), names="value")
    display(slider)

    return fig, axs
