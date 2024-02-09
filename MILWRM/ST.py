# -*- coding: utf-8 -*-
"""
Functions and classes for manipulating 10X Visium spatial transcriptomic (ST) and 
histological imaging data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
import squidpy as sq

sc.set_figure_params(dpi=100, dpi_save=400)
sns.set_style("white")
plt.rcParams["font.family"] = "sans.serif"

from math import ceil
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.interpolate import interpnd, griddata
from sklearn.metrics.pairwise import euclidean_distances


def blur_features_st(adata, tmp, spatial_graph_key=None, n_rings=1):
    """
    Blur values in an `AnnData` object using spatial nearest neighbors

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing Visium data
    tmp : pd.DataFrame
        containing feature columns from adata.obs that will be blurred
    spatial_graph_key : str, optional (default=`None`)
        Key in `adata.obsp` containing spatial graph connectivities (i.e.
        `"spatial_connectivities"`). If `None`, compute new spatial graph using
        `n_rings` in `squidpy`.
    n_rings : int, optional (default=1)
        Number of hexagonal rings around each spatial transcriptomics spot to blur
        features by for capturing regional information. Assumes 10X Genomics Visium
        platform.

    Returns
    -------
    adata.obs is edited in place with new blurred columns
    """
    if spatial_graph_key is not None:
        # use existing spatial graph
        assert (
            spatial_graph_key in adata.obsp.keys()
        ), "Spatial connectivities key '{}' not found.".format(spatial_graph_key)
    else:
        # create spatial graph
        print("Computing spatial graph with {} hexagonal rings".format(n_rings))
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_rings=n_rings)
        spatial_graph_key = "spatial_connectivities"  # set key to expected output
    tmp2 = tmp.copy()  # copy of temporary dataframe for dropping blurred features into
    cols = tmp.columns  # get column names
    # perform blurring by nearest spot neighbors
    for x in range(len(tmp)):
        vals = tmp.iloc[
            list(
                np.argwhere(
                    adata.obsp[spatial_graph_key][
                        x,
                    ]
                )[:, 1]
            )
            + [x],
            :,
        ].mean()
        tmp2.iloc[x, :] = vals.values
    # add blurred features to anndata object
    adata.obs[[x for x in cols]] = tmp.loc[:, cols].values
    adata.obs[["blur_" + x for x in cols]] = tmp2.loc[:, cols].values
    return tmp2.loc[:, cols]


def bin_threshold(mat, threshmin=None, threshmax=0.5):
    """
    Generate binary segmentation from probabilities

    Parameters
    ----------
    mat : np.array
        The data
    threshmin : float or None
        Minimum value on [0,1] to assign binary IDs from probabilities.
    thresmax : float
        Maximum value on [0,1] to assign binary IDs from probabilities. Values higher
        than threshmax -> 1. Values lower than thresmax -> 0.

    Returns
    -------
    a : np.array
        Thresholded matrix
    """
    a = np.ma.array(mat, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin:
        mask |= (a < threshmin).filled(False)

    if threshmax:
        mask |= (a > threshmax).filled(False)

    a[mask] = 1
    a[~mask] = 0
    return a


def map_pixels(
    adata,
    filter_label="in_tissue",
    img_key="hires",
    library_id=None,
    map_size=None,
):
    """
    Map spot IDs to 'pixel space' by assigning spot ID values to evenly spaced grid

    Parameters
    ----------
    adata : AnnData.anndata
        The data
    filter_label : str or None
        adata.obs column key that contains binary labels for filtering barcodes. If
        None, do not filter.
    img_key : str
        adata.uns key containing the image to use for mapping
    library_id : str, optional (default=None)
        Key for finding proper library from adata.uns["spatial"]. By default, find
        the key from adata.uns["spatial"].keys()
    map_size : tuple of int, optional (default=None)
        Shape of image to map to. By default, trim to ST coordinates. Can provide
        shape of whole hires image in adata.uns["spatial"] to yield pitas at full
        H&E image size.

    Returns
    -------
    adata : AnnData.anndata
        with the following attributes:
        adata.uns["pixel_map_df"] : pd.DataFrame
            Long-form dataframe of Visium spot barcode IDs, pixel coordinates, and
            .obs metadata
        adata.uns["pixel_map"] : np.array
            Pixel space array of Visium spot barcode IDs
    """
    adata.uns["pixel_map_params"] = {
        "img_key": img_key
    }  # create params dict for future use
    # add library_id key to params
    if library_id is None:
        library_id = adata.uns["pixel_map_params"]["library_id"] = list(
            adata.uns["spatial"].keys()
        )[0]
    else:
        adata.uns["pixel_map_params"]["library_id"] = library_id
    # first get center-to-face pixel distance of hexagonal Visium spots
    dist = euclidean_distances(adata.obsm["spatial"])
    adata.uns["pixel_map_params"]["ctr_to_face"] = (
        np.unique(dist)[np.unique(dist) != 0].min() / 2
    )
    # also save center-to-vertex pixel distance as vadata attribute
    adata.uns["pixel_map_params"]["ctr_to_vert"] = adata.uns["pixel_map_params"][
        "ctr_to_face"
    ] / np.cos(30 * (np.pi / 180))
    # get the spot radius from adata.uns["spatial"] as well
    adata.uns["pixel_map_params"]["radius"] = (
        adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] / 2
    )
    # get scale factor from adata.uns["spatial"]
    adata.uns["pixel_map_params"]["scalef"] = adata.uns["spatial"][library_id][
        "scalefactors"
    ][f"tissue_{img_key}_scalef"]

    if filter_label is not None:
        # create frame of mock pixels to make edges look better
        # x and y deltas for moving rows and columns into a blank frame
        delta_x = (
            adata[adata.obs.array_col == 0, :].obsm["spatial"]
            - adata[adata.obs.array_col == 1, :].obsm["spatial"]
        )
        delta_x = np.mean(list(delta_x[:, 1])) * 2
        delta_y = (
            adata[adata.obs.array_row == 0, :].obsm["spatial"]
            - adata[adata.obs.array_row == 1, :].obsm["spatial"]
        )
        delta_y = np.mean(list(delta_y[:, 1])) * 2
        # left part of frame, translated
        left = adata[
            adata.obs.array_col.isin(
                [adata.obs.array_col.max() - 2, adata.obs.array_col.max() - 3]
            ),
            :,
        ].copy()
        left.obsm["spatial"][..., 0] -= delta_x.astype(int)
        del left.var
        del left.uns
        left.obs[filter_label] = 0
        left.obs_names = ["left" + str(x) for x in range(left.n_obs)]
        # right part of frame, translated
        right = adata[adata.obs.array_col.isin([2, 3]), :].copy()
        right.obsm["spatial"][..., 0] += delta_x.astype(int)
        del right.var
        del right.uns
        right.obs[filter_label] = 0
        right.obs_names = ["right" + str(x) for x in range(right.n_obs)]
        # add sides to orig
        a_sides = adata.concatenate(
            [left, right],
            index_unique=None,
        )
        a_sides.obs.drop(columns="batch", inplace=True)
        # bottom part of frame, translated
        bottom = a_sides[a_sides.obs.array_row == 1, :].copy()
        bottom.obsm["spatial"][..., 1] += delta_y.astype(int)
        bottom.obs_names = ["bottom" + str(x) for x in range(bottom.n_obs)]
        del bottom.var
        del bottom.uns
        bottom.obs[filter_label] = 0
        # top part of frame, translated
        top = a_sides[
            a_sides.obs.array_row == a_sides.obs.array_row.max() - 1, :
        ].copy()
        top.obsm["spatial"][..., 1] -= delta_y.astype(int)
        del top.var
        del top.uns
        top.obs[filter_label] = 0
        top.obs_names = ["top" + str(x) for x in range(top.n_obs)]
        # complete frame
        a_frame = a_sides.concatenate(
            [top, bottom],
            index_unique=None,
        )
        a_frame.uns = adata.uns
        a_frame.var = adata.var
        a_frame.obs.drop(columns="batch", inplace=True)
    else:
        a_frame = adata.copy()

    # determine pixel bounds from spot coords, adding center-to-face distance
    a_frame.uns["pixel_map_params"]["xmin_px"] = int(
        np.floor(
            a_frame.uns["pixel_map_params"]["scalef"]
            * (
                a_frame.obsm["spatial"][:, 0].min()
                - a_frame.uns["pixel_map_params"]["radius"]
            )
        )
    )
    a_frame.uns["pixel_map_params"]["xmax_px"] = int(
        np.ceil(
            a_frame.uns["pixel_map_params"]["scalef"]
            * (
                a_frame.obsm["spatial"][:, 0].max()
                + a_frame.uns["pixel_map_params"]["radius"]
            )
        )
    )
    a_frame.uns["pixel_map_params"]["ymin_px"] = int(
        np.floor(
            a_frame.uns["pixel_map_params"]["scalef"]
            * (
                a_frame.obsm["spatial"][:, 1].min()
                - a_frame.uns["pixel_map_params"]["radius"]
            )
        )
    )
    a_frame.uns["pixel_map_params"]["ymax_px"] = int(
        np.ceil(
            a_frame.uns["pixel_map_params"]["scalef"]
            * (
                a_frame.obsm["spatial"][:, 1].max()
                + a_frame.uns["pixel_map_params"]["radius"]
            )
        )
    )

    print("Creating pixel grid and mapping to nearest barcode coordinates")
    # define grid for pixel space
    if map_size is not None:
        # use provided size
        assert (
            map_size[1]
            >= a_frame.uns["pixel_map_params"]["ymax_px"]
            - a_frame.uns["pixel_map_params"]["ymin_px"]
        ), "Given map_size isn't large enough."
        assert (
            map_size[0]
            >= a_frame.uns["pixel_map_params"]["xmax_px"]
            - a_frame.uns["pixel_map_params"]["xmin_px"]
        ), "Given map_size isn't large enough."
        grid_y, grid_x = np.mgrid[
            0 : map_size[0],
            0 : map_size[1],
        ]
        # set min and max pixels to map_size
        a_frame.uns["pixel_map_params"]["ymin_px"] = 0
        a_frame.uns["pixel_map_params"]["ymax_px"] = map_size[0]
        a_frame.uns["pixel_map_params"]["xmin_px"] = 0
        a_frame.uns["pixel_map_params"]["xmax_px"] = map_size[1]

    else:
        # determine size from a.obsm["spatial"]
        grid_y, grid_x = np.mgrid[
            a_frame.uns["pixel_map_params"]["ymin_px"] : a_frame.uns[
                "pixel_map_params"
            ]["ymax_px"],
            a_frame.uns["pixel_map_params"]["xmin_px"] : a_frame.uns[
                "pixel_map_params"
            ]["xmax_px"],
        ]

    # map barcodes to pixel coordinates
    pixel_coords = np.column_stack((grid_x.ravel(order="C"), grid_y.ravel(order="C")))
    barcode_list = griddata(
        np.multiply(a_frame.obsm["spatial"], a_frame.uns["pixel_map_params"]["scalef"]),
        a_frame.obs_names,
        (pixel_coords[:, 0], pixel_coords[:, 1]),
        method="nearest",
    )
    # save grid_x and grid_y to adata.uns
    a_frame.uns["grid_x"], a_frame.uns["grid_y"] = grid_x, grid_y

    # put results into DataFrame for filtering and reindexing
    print("Saving barcode mapping to adata.uns['pixel_map_df'] and adding metadata")
    a_frame.uns["pixel_map_df"] = pd.DataFrame(pixel_coords, columns=["x", "y"])
    # add barcodes to long-form dataframe
    a_frame.uns["pixel_map_df"]["barcode"] = barcode_list
    # merge master df with self.adata.obs for metadata
    a_frame.uns["pixel_map_df"] = a_frame.uns["pixel_map_df"].merge(
        a_frame.obs, how="outer", left_on="barcode", right_index=True
    )
    # filter using label from adata.obs if desired (i.e. "in_tissue")
    if filter_label is not None:
        print(
            "Filtering barcodes using labels in self.adata.obs['{}']".format(
                filter_label
            )
        )
        # set empty pixels (no Visium spot) to "none"
        a_frame.uns["pixel_map_df"].loc[
            a_frame.uns["pixel_map_df"][filter_label] == 0,
            "barcode",
        ] = "none"
        # subset the entire anndata object using filter_label
        a_frame = a_frame[a_frame.obs[filter_label] == 1, :].copy()
        print("New size: {} spots x {} genes".format(a_frame.n_obs, a_frame.n_vars))

    print("Done!")
    return a_frame


def trim_image(
    adata, distance_trim=False, threshold=None, channels=None, plot_out=True, **kwargs
):
    """
    Trim pixels in image using pixel map output from Visium barcodes

    Parameters
    ----------
    adata : AnnData.anndata
        The data
    distance_trim : bool
        Manually trim pixels by distance to nearest Visium spot center
    threshold : int or None
        Number of pixels from nearest Visium spot center to call barcode ID. Ignored
        if `distance_trim==False`.
    channels : list of str or None
        Names of image channels in axis order. If None, channels are named "ch_0",
        "ch_1", etc.
    plot_out : bool
        Plot final trimmed image
    **kwargs
        Arguments to pass to `show_pita()` function if `plot_out==True`

    Returns
    -------
    adata.uns["pixel_map_trim"] : np.array
        Contains image with unused pixels set to `np.nan`
    adata.obsm["spatial_trim"] : np.array
        Contains spatial coords with adjusted pixel values after image cropping
    """
    assert (
        adata.uns["pixel_map_params"] is not None
    ), "Pixel map not yet created. Run map_pixels() first."

    print(
        "Cropping image to pixel dimensions and adding values to adata.uns['pixel_map_df']"
    )
    cropped = adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
        "images"
    ][adata.uns["pixel_map_params"]["img_key"]].transpose(1, 0, 2)[
        int(adata.uns["pixel_map_params"]["xmin_px"]) : int(
            (adata.uns["pixel_map_params"]["xmax_px"])
        ),
        int(adata.uns["pixel_map_params"]["ymin_px"]) : int(
            (adata.uns["pixel_map_params"]["ymax_px"])
        ),
    ]
    # crop x,y coords and save to .obsm as well
    print("Cropping Visium spot coordinates and saving to adata.obsm['spatial_trim']")
    adata.obsm["spatial_trim"] = adata.obsm["spatial"] - np.repeat(
        [
            [
                adata.uns["pixel_map_params"]["xmin_px"],
                adata.uns["pixel_map_params"]["ymin_px"],
            ]
        ],
        adata.obsm["spatial"].shape[0],
        axis=0,
    )

    # manual trimming of pixels by distance if desired
    if distance_trim:
        print("Calculating pixel distances from spot centers for thresholding")
        tree = cKDTree(adata.obsm["spatial"])
        xi = interpnd._ndim_coords_from_arrays(
            (adata.uns["grid_x"], adata.uns["grid_y"]),
            ndim=adata.obsm["spatial"].shape[1],
        )
        dists, _ = tree.query(xi)

        # determine distance threshold
        if threshold is None:
            threshold = int(adata.uns["pixel_map_params"]["ctr_to_vert"] + 1)
            print(
                "Using distance threshold of {} pixels from adata.uns['pixel_map_params']['ctr_to_vert']".format(
                    threshold
                )
            )

        dist_mask = bin_threshold(dists, threshmax=threshold)
        if plot_out:
            # plot pixel distances from spot centers on image
            show_pita(pita=dists, figsize=(4, 4))
            # plot binary thresholded image
            show_pita(pita=dist_mask, figsize=(4, 4))

        print(
            "Trimming pixels by spot distance and adjusting labels in adata.uns['pixel_map_df']"
        )
        mask_df = pd.DataFrame(dist_mask.T.ravel(order="F"), columns=["manual_trim"])
        adata.uns["pixel_map_df"] = adata.uns["pixel_map_df"].merge(
            mask_df, left_index=True, right_index=True
        )
        adata.uns["pixel_map_df"].loc[
            adata.uns["pixel_map_df"]["manual_trim"] == 1, ["barcode"]
        ] = "none"  # set empty pixels to empty barcode
        adata.uns["pixel_map_df"].drop(
            columns="manual_trim", inplace=True
        )  # remove unneeded label

    if channels is None:
        # if channel names not specified, name them numerically
        channels = ["ch_{}".format(x) for x in range(cropped.shape[2])]
    # cast image intensity values to long-form and add to adata.uns["pixel_map_df"]
    rgb = pd.DataFrame(
        np.column_stack(
            [cropped[:, :, x].ravel(order="F") for x in range(cropped.shape[2])]
        ),
        columns=channels,
    )
    adata.uns["pixel_map_df"] = adata.uns["pixel_map_df"].merge(
        rgb, left_index=True, right_index=True
    )
    adata.uns["pixel_map_df"].loc[
        adata.uns["pixel_map_df"]["barcode"] == "none", channels
    ] = np.nan  # set empty pixels to invalid image intensity value

    # calculate mean image values for each channel and create .obsm key
    adata.obsm["image_means"] = (
        adata.uns["pixel_map_df"]
        .loc[adata.uns["pixel_map_df"]["barcode"] != "none", ["barcode"] + channels]
        .groupby("barcode")
        .mean()
        .values
    )

    print(
        "Saving cropped and trimmed image to adata.uns['spatial']['{}']['images']['{}_trim']".format(
            adata.uns["pixel_map_params"]["library_id"],
            adata.uns["pixel_map_params"]["img_key"],
        )
    )
    adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]]["images"][
        "{}_trim".format(adata.uns["pixel_map_params"]["img_key"])
    ] = np.dstack(
        [
            adata.uns["pixel_map_df"]
            .pivot(index="y", columns="x", values=[channels[x]])
            .values
            for x in range(len(channels))
        ]
    )
    # save scale factor as well
    adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]]["scalefactors"][
        "tissue_{}_trim_scalef".format(adata.uns["pixel_map_params"]["img_key"])
    ] = adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
        "scalefactors"
    ][
        "tissue_{}_scalef".format(adata.uns["pixel_map_params"]["img_key"])
    ]
    # plot results if desired
    if plot_out:
        if len(channels) == 3:
            show_pita(
                pita=adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
                    "images"
                ]["{}_trim".format(adata.uns["pixel_map_params"]["img_key"])],
                RGB=True,
                label=channels,
                **kwargs,
            )
        else:
            show_pita(
                pita=adata.uns["spatial"][adata.uns["pixel_map_params"]["library_id"]][
                    "images"
                ]["{}_trim".format(adata.uns["pixel_map_params"]["img_key"])],
                RGB=False,
                label=channels,
                **kwargs,
            )
    print("Done!")


def assemble_pita(
    adata,
    features=None,
    use_rep=None,
    layer=None,
    plot_out=True,
    histo=None,
    verbose=True,
    **kwargs,
):
    """
    Cast feature into pixel space to construct gene expression image ("pita")

    Parameters
    ----------
    adata : AnnData.anndata
        the data
    features : list of int or str
        Names or indices of features to cast onto spot image. If `None`, cast all
        features. If `plot_out`, first feature in list will be plotted. If not
        specified and `plot_out`, first feature (index 0) will be plotted.
    use_rep : str
        Key from `adata.obsm` to use for plotting. If `None`, use `adata.X`.
    layer :str
        Key from `adata.layers` to use for plotting. Ignored if `use_rep` is not `None`
    plot_out : bool
        Show resulting image?
    histo : str or `None`, optional (default=`None`)
        Histology image to show along with pita in gridspec (i.e. "hires",
        "hires_trim", "lowres"). If `None` or if `plot_out`==`False`, ignore.
    verbose : bool, optional (default=`True`)
        Print updates to console
    **kwargs
        Arguments to pass to `show_pita()` function

    Returns
    -------
    assembled : np.array
        Image of desired expression in pixel space
    """
    assert (
        adata.uns["pixel_map_params"] is not None
    ), "Pixel map not yet created. Run map_pixels() first."

    # coerce features to list if only single string
    if features and not isinstance(features, list):
        features = [features]

    if use_rep is None:
        # use all genes if no gene features specified
        if not features:
            features = adata.var_names  # [adata.var.highly_variable == 1].tolist()
        if layer is None:
            if verbose:
                print(
                    "Assembling pita with {} features from adata.X".format(
                        len(features)
                    )
                )
            mapper = pd.DataFrame(
                adata.X[:, [adata.var_names.get_loc(x) for x in features]],
                index=adata.obs_names,
            )
        else:
            if verbose:
                print(
                    "Assembling pita with {} features from adata.layers['{}']".format(
                        len(features), layer
                    )
                )
            mapper = pd.DataFrame(
                adata.layers[layer][:, [adata.var_names.get_loc(x) for x in features]],
                index=adata.obs_names,
            )
    elif use_rep in [".obs", "obs"]:
        assert features is not None, "Must provide feature(s) from adata.obs"
        if verbose:
            print(
                "Assembling pita with {} features from adata.obs".format(len(features))
            )
        if all(isinstance(x, int) for x in features):
            mapper = adata.obs.iloc[:, features].copy()
        else:
            mapper = adata.obs[features].copy()
    else:
        if not features:
            if verbose:
                print(
                    "Assembling pita with {} features from adata.obsm['{}']".format(
                        adata.obsm[use_rep].shape[1], use_rep
                    )
                )
            mapper = pd.DataFrame(adata.obsm[use_rep], index=adata.obs_names)
        else:
            assert all(
                isinstance(x, int) for x in features
            ), "Features must be integer indices if using rep from adata.obsm"
            if verbose:
                print(
                    "Assembling pita with {} features from adata.obsm['{}']".format(
                        len(features), use_rep
                    )
                )
            mapper = pd.DataFrame(
                adata.obsm[use_rep][:, features], index=adata.obs_names
            )

    # check for categorical columns to force into discrete plots
    discrete_cols = {}
    for col in mapper.columns:
        if pd.api.types.is_categorical_dtype(mapper[col]):
            cat_max = len(mapper[col].cat.categories)
            categories = mapper[col].cat.categories  # save original categories
            mapper[col] = mapper[col].replace(
                {v: k for k, v in dict(enumerate(mapper[col].cat.categories)).items()}
            )
            discrete_cols[mapper.columns.get_loc(col)] = (cat_max, categories)
    # if no categorical columns, pass None to discrete_cols
    if bool(discrete_cols) is False:
        discrete_cols = None

    # cast barcodes into pixel dimensions for reindexing
    if verbose:
        print(
            "Casting barcodes to pixel dimensions and saving to adata.uns['pixel_map']"
        )
    pixel_map = (
        adata.uns["pixel_map_df"].pivot(index="y", columns="x", values="barcode").values
    )

    assembled = np.array(
        [mapper.reindex(index=pixel_map[x], copy=True) for x in range(len(pixel_map))]
    ).squeeze()

    if plot_out:
        # determine where the histo image is in anndata
        if histo is not None:
            assert (
                histo
                in adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                    "images"
                ].keys()
            ), "Must provide one of {} for histo".format(
                adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                    "images"
                ].keys()
            )
            histo = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]][
                "images"
            ][histo]
        show_pita(
            pita=assembled,
            features=None,
            discrete_features=discrete_cols,
            histo=histo,
            **kwargs,
        )
    if verbose:
        print("Done!")
    return assembled, discrete_cols


def plot_single_image(
    image,
    ax,
    label="",
    cmap="plasma",
    **kwargs,
):
    """
    Plot a pixel image

    Parameters
    ----------
    image : np.array
        Image to plot
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot `image` to
    label : str, optional (default="")
        What to title the image plot
    cmap : str, optional (default="plasma")
        Matplotlib colormap to use
    **kwargs
        Arguments to pass to `plt.imshow()` function

    Returns
    -------
    Matplotlib axes containing plot of image with associated colorbar
    """
    assert image.ndim > 1, "Image does not have enough dimensions: {} given".format(
        image.ndim
    )
    assert image.ndim < 3, "Image has too many dimensions: {} given".format(image.ndim)
    # call imshow with discrete colormap for categorical plot
    im = ax.imshow(image, cmap=plt.cm.get_cmap(cmap), **kwargs)
    # clean up axes
    plt.tick_params(labelbottom=False, labelleft=False)
    sns.despine(bottom=True, left=True)
    # title above plot
    plt.title(
        label=label,
        loc="left",
        fontweight="bold",
        fontsize=16,
    )
    _ = plt.colorbar(im, shrink=0.7, ticks=None)


def plot_single_image_discrete(
    image,
    ax,
    max_val,
    ticklabels=None,
    label="",
    cmap="plasma",
    **kwargs,
):
    """
    Plot a discrete (categorical) pixel image containing integer values (i.e. MILWRM
    domains)

    Parameters
    ----------
    image : np.array
        Image to plot containing zero-indexed, integer values per pixel
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot `image` to
    max_val : int
        Maximum integer value for categories (i.e. 4 for categories [0,1,2,3,4]).
        Categories are expected to be zero-indexed integers.
    ticklabels : list of str, optional (default=`None`)
        Ordered list of categories for labeling discrete colorbar ticks. If `None`,
        number categories 0 - `max_val`.
    label : str, optional (default="")
        What to title the image plot
    cmap : str, optional (default="plasma")
        Matplotlib colormap to use
    **kwargs
        Arguments to pass to `plt.imshow()` function

    Returns
    -------
    Matplotlib axes containing discrete plot of image with associated colorbar
    """
    assert image.ndim > 1, "Image does not have enough dimensions: {} given".format(
        image.ndim
    )
    assert image.ndim < 3, "Image has too many dimensions: {} given".format(image.ndim)
    # call imshow with discrete colormap for categorical plot
    im = ax.imshow(image, cmap=plt.cm.get_cmap(cmap, int(max_val)), **kwargs)
    # clean up axes
    plt.tick_params(labelbottom=False, labelleft=False)
    sns.despine(bottom=True, left=True)
    # title above plot
    plt.title(
        label=label,
        loc="left",
        fontweight="bold",
        fontsize=16,
    )
    cbar = plt.colorbar(im, shrink=0.7, ticks=range(int(max_val)))
    # move edges of colorbar by 0.5
    im.set_clim(vmin=-0.5, vmax=max_val - 0.5)
    # add custom ticklabels based on categories
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)


def plot_single_image_rgb(
    image,
    ax,
    channels=None,
    label="",
    **kwargs,
):
    """
    Plot an RGB pixel image

    Parameters
    ----------
    image : np.array
        3-dimensional image to plot of shape (n, m, 3)
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot `image` to
    channels : list of str or None, optional (default=`None`)
        List of channel names in order of (R,G,B) for legend. If `None`, no legend.
    label : str, optional (default="")
        What to title the image plot
    **kwargs
        Arguments to pass to `plt.imshow()` function

    Returns
    -------
    Matplotlib axes containing plot of image with associated RGB legend
    """
    assert (image.ndim == 3) & (
        image.shape[2] == 3
    ), "Need 3 dimensions and 3 given features for an RGB image; shape = {}".format(
        image.shape
    )
    # call imshow
    _ = ax.imshow(image, **kwargs)
    if channels is not None:
        # add legend for channel IDs
        custom_lines = [
            Line2D([0], [0], color=(1, 0, 0), lw=5),
            Line2D([0], [0], color=(0, 1, 0), lw=5),
            Line2D([0], [0], color=(0, 0, 1), lw=5),
        ]
        # custom RGB legend
        plt.legend(
            custom_lines,
            channels,
            fontsize="medium",
            bbox_to_anchor=(1, 1),
            loc="upper left",
        )
    # clean up axes
    plt.tick_params(labelbottom=False, labelleft=False)
    sns.despine(bottom=True, left=True)
    # title above plot
    plt.title(
        label=label,
        loc="left",
        fontweight="bold",
        fontsize=16,
    )


def show_pita(
    pita,
    features=None,
    discrete_features=None,
    RGB=False,
    histo=None,
    label="feature",
    ncols=4,
    figsize=(7, 7),
    cmap="plasma",
    save_to=None,
    **kwargs,
):
    """
    Plot assembled pita using `plt.imshow()`

    Parameters
    ----------
    pita : np.array
        Image of desired expression in pixel space from `.assemble_pita()`
    features : list of int, optional (default=`None`)
        List of features by index to show in plot. If `None`, use all features.
    discrete_features : dict, optional (default=`None`)
        Dictionary of feature indices (keys) containing discrete (categorical) values
        (i.e. MILWRM domain). Values are tuple of `max_value` to pass to
        `plot_single_image_discrete` for each discrete feature, and the ordered list
        of categories for legend plotting. If `None`, treat all features as continuous.
    RGB : bool, optional (default=`False`)
        Treat 3-dimensional array as RGB image
    histo : np.array or `None`, optional (default=`None`)
        Histology image to show along with pita in gridspec. If `None`, ignore.
    label : str, optional (default="feature")
        What to title each panel of the gridspec (i.e. "PC" or "usage") or each
        channel in RGB image. Can also pass list of names e.g. ["NeuN","GFAP",
        "DAPI"] corresponding to channels.
    ncols : int, optional (default=4)
        Number of columns for gridspec
    figsize : tuple of float, optional (default=(7, 7))
        Size in inches of output figure
    cmap : str, optional (default="plasma")
        Matplotlib colormap to use
    save_to : str or None, optional (default=`None`)
        Path to image file to save results. if `None`, show figure.
    **kwargs
        Arguments to pass to `plt.imshow()` function

    Returns
    -------
    Matplotlib object (if plotting one feature or RGB) or gridspec object (for
    multiple features). Saves plot to file if `save_to` is not `None`.
    """
    assert pita.ndim > 1, "Pita does not have enough dimensions: {} given".format(
        pita.ndim
    )
    assert pita.ndim < 4, "Pita has too many dimensions: {} given".format(pita.ndim)
    # if only one feature (2D), plot it quickly
    if (pita.ndim == 2) and histo is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if discrete_features is not None:
            plot_single_image_discrete(
                image=pita,
                ax=ax,
                # use first value in dict as max
                max_val=list(discrete_features.values())[0][0],
                ticklabels=list(discrete_features.values())[0][1],
                label=label[0] if isinstance(label, list) else label,
                cmap=cmap,
                **kwargs,
            )
        else:
            plot_single_image(
                image=pita,
                ax=ax,
                label=label[0] if isinstance(label, list) else label,
                cmap=cmap,
                **kwargs,
            )
        plt.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig
    if (pita.ndim == 2) and histo is not None:
        n_rows, n_cols = 1, 2  # two images here, histo and RGB
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        ax = plt.subplot(gs[0])
        plot_single_image_rgb(
            image=histo,
            ax=ax,
            channels=None,
            label="Histology",
            **kwargs,
        )
        ax = plt.subplot(gs[1])
        if discrete_features is not None:
            plot_single_image_discrete(
                image=pita,
                ax=ax,
                # use first value in dict as max
                max_val=list(discrete_features.values())[0][0],
                ticklabels=list(discrete_features.values())[0][1],
                label=label[0] if isinstance(label, list) else label,
                cmap=cmap,
                **kwargs,
            )
        else:
            plot_single_image(
                image=pita,
                ax=ax,
                label=label[0] if isinstance(label, list) else label,
                cmap=cmap,
                **kwargs,
            )
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig
    if RGB:
        # if third dim has 3 features, treat as RGB and plot it quickly
        assert (pita.ndim == 3) & (
            pita.shape[2] == 3
        ), "Need 3 dimensions and 3 given features for an RGB image; shape = {}; features given = {}".format(
            pita.shape, len(features)
        )
        print("Plotting pita as RGB image")
        if isinstance(label, str):
            # if label is single string, name channels numerically
            channels = ["{}_{}".format(label, x) for x in range(pita.shape[2])]
        else:
            assert (
                len(label) == 3
            ), "Please pass 3 channel names for RGB plot; {} labels given: {}".format(
                len(label), label
            )
            channels = label
        if histo is not None:
            n_rows, n_cols = 1, 2  # two images here, histo and RGB
            fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
            # arrange axes as subplots
            gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
            # add plots to axes
            ax = plt.subplot(gs[0])
            plot_single_image_rgb(
                image=histo,
                ax=ax,
                channels=None,
                label="Histology",
                **kwargs,
            )
            ax = plt.subplot(gs[1])
            plot_single_image_rgb(
                image=pita,
                ax=ax,
                channels=channels,
                label="",
                **kwargs,
            )
            fig.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            plot_single_image_rgb(
                image=pita,
                ax=ax,
                channels=channels,
                label="",
                **kwargs,
            )
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=300
                )
            return fig
    # if pita has multiple features, plot them in gridspec
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    # if no features are given, use all of them
    if features is None:
        features = [x for x in range(pita.shape[2])]
    else:
        assert (
            pita.ndim > 2
        ), "Not enough features in pita: shape {}, expecting 3rd dim with length {}".format(
            pita.shape, len(features)
        )
        assert (
            len(features) <= pita.shape[2]
        ), "Too many features given: pita has {}, expected {}".format(
            pita.shape[2], len(features)
        )
    if isinstance(label, str):
        # if label is single string, name channels numerically
        labels = ["{}_{}".format(label, x) for x in features]
    else:
        assert len(label) == len(
            features
        ), "Please provide the same number of labels as features; {} labels given, {} features given.".format(
            len(label), len(features)
        )
        labels = label
    # calculate gridspec dimensions
    if histo is not None:
        labels = ["Histology"] + labels  # append histo to front of labels
        if len(features) + 1 <= ncols:
            n_rows, n_cols = 1, len(features) + 1
        else:
            n_rows, n_cols = ceil((len(features) + 1) / ncols), ncols
    else:
        if len(features) <= ncols:
            n_rows, n_cols = 1, len(features)
        else:
            n_rows, n_cols = ceil(len(features) / ncols), ncols
    fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
    # arrange axes as subplots
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    # add plots to axes
    i = 0
    if histo is not None:
        # add histology plot to first axes
        ax = plt.subplot(gs[i])
        plot_single_image_rgb(
            image=histo,
            ax=ax,
            channels=None,
            label=labels[i],
            **kwargs,
        )
        i = i + 1
    for feature in features:
        ax = plt.subplot(gs[i])
        if discrete_features is not None:
            if feature in discrete_features:
                plot_single_image_discrete(
                    image=pita[:, :, feature],
                    ax=ax,
                    # use corresponding value in dict as max
                    max_val=discrete_features[feature][0],
                    ticklabels=discrete_features[feature][1],
                    label=labels[i],
                    cmap=cmap,
                    **kwargs,
                )
            else:
                plot_single_image(
                    image=pita[:, :, feature],
                    ax=ax,
                    label=labels[i],
                    cmap=cmap,
                    **kwargs,
                )
        else:
            plot_single_image(
                image=pita[:, :, feature],
                ax=ax,
                label=labels[i],
                cmap=cmap,
                **kwargs,
            )
        i = i + 1
    fig.tight_layout()
    if save_to:
        plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
    return fig
