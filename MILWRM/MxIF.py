# -*- coding: utf-8 -*-
"""
Functions and classes for analyzing multiplex imaging data
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

sns.set_style("white")
plt.rcParams["font.family"] = "sans.serif"

from math import ceil
from skimage import exposure
from skimage.io import imread
from skimage.measure import block_reduce
from matplotlib.lines import Line2D
from skimage import filters
from skimage.restoration import denoise_bilateral


def checktype(obj):
    return bool(obj) and all(isinstance(elem, str) for elem in obj)


def clip_values(image, channels=None):
    """
    Clip outlier values from specified channels of an image

    Parameters
    ----------
    image : np.ndarray
        The image
    channels : tuple of int or None, optional (default=`None`)
        Channels to clip on img.shape[2]. If None, clip values in all channels.

    Returns
    -------
    image_cp : np.ndarray
        Image with clipped values
    """
    image_cp = image.copy()
    if channels is None or image.ndim == 2:
        vmin, vmax = np.nanpercentile(image_cp[image_cp != -99999], q=(0.5, 99.5))
        plane_clip = exposure.rescale_intensity(
            image_cp,
            in_range=(vmin, vmax),
            out_range=np.float32,
        )
        image_cp = plane_clip
    else:
        for z in channels:
            plane = image_cp[:, :, z].copy()
            vmin, vmax = np.nanpercentile(plane, q=(0.5, 99.5))
            plane_clip = exposure.rescale_intensity(
                plane,
                in_range=(vmin, vmax),
                out_range=np.float32,
            )
            image_cp[:, :, z] = plane_clip
    return image_cp


def scale_rgb(image, channels=None):
    """
    Scale to [0.0, 1.0] for RGB image

    Parameters
    ----------
    image : np.ndarray
        The image
    channels : tuple of int or None, optional (default=`None`)
        Channels to scale on img.shape[2]. If None, scale values in all channels.

    Returns
    -------
    image_cp : np.ndarray
        Image with scaled values
    """
    image_cp = image.copy()
    if channels is None or image.ndim == 2:
        image_cp = image_cp - image_cp.min()
        image_cp = image_cp / image_cp.max()
    else:
        for z in channels:
            plane = image_cp[:, :, z].copy()
            plane = plane - plane.min()
            image_cp[:, :, z] = plane / plane.max()
    return image_cp


def CLAHE(image, channels=None, **kwargs):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE)

    Parameters
    ----------
    image : np.ndarray
        The image
    channels : tuple of int or None, optional (default=`None`)
        Channels to adjust on image.shape[2]. If None, perform CLAHE in all channels.
    **kwargs
        Keyword arguments to pass to `skimage.exposure.equalize_adapthist`

    Returns
    -------
    image_cp : np.ndarray
        Image with exposure-adjusted values
    """
    image_cp = image.copy()
    if image.ndim == 2:
        image_cp = exposure.equalize_adapthist(image_cp, **kwargs)
    elif channels is None:
        for z in range(image_cp.shape[2]):
            image_cp[:, :, z] = exposure.equalize_adapthist(image_cp[:, :, z], **kwargs)
    else:
        for z in channels:
            image_cp[:, :, z] = exposure.equalize_adapthist(image_cp[:, :, z], **kwargs)
    return image_cp


class img:
    def __init__(self, img_arr, channels=None, mask=None):
        """
        Initialize img class

        Parameters
        ----------
        img_arr : np.ndarray
            The image as a numpy array
        channels : list of str or None, optional (default=`None`)
            List of channel names corresponding to img.shape[2]. i.e. `("DAPI","GFAP",
            "NeuH")`. If `None`, channels are named "ch_0", "ch_1", etc.
        mask : np.ndarray
            Mask defining pixels containing tissue in the image

        Returns
        -------
        `img` object
        """
        assert (
            img_arr.ndim > 1
        ), "Image does not have enough dimensions: {} given".format(img_arr.ndim)
        self.img = img_arr.astype("float64")  # save image array to .img attribute
        if img_arr.ndim > 2:
            self.n_ch = img_arr.shape[2]  # save number of channels to attribute
        else:
            self.n_ch = 1
        if channels is None:
            # if channel names not specified, name them numerically
            self.ch = ["ch_{}".format(x) for x in range(self.n_ch)]
        else:
            if not isinstance(channels, list):
                raise Exception("Channels must be given in a list")
            assert (
                len(channels) == self.n_ch
            ), "Number of channels must match img_arr.shape[2]"
            self.ch = channels
        if mask is not None:
            # validate that mask matches img_arr
            assert (
                mask.shape == img_arr.shape[:2]
            ), "Shape of mask must match the first two dimensions of img_arr"
        self.mask = mask  # set mask attribute, regardless of value given

    def __repr__(self) -> str:
        """Representation of contents of img object"""
        descr = "img object with {} of {} and shape {}px x {}px\n".format(
            type(self.img),
            self.img.dtype,
            self.img.shape[0],
            self.img.shape[1],
        ) + "{} image channels:\n\t{}".format(self.n_ch, self.ch)
        if self.mask is not None:
            descr += "\n\ntissue mask {} of {} and shape {}px x {}px".format(
                type(self.mask),
                self.mask.dtype,
                self.mask.shape[0],
                self.mask.shape[1],
            )
        return descr

    def copy(self) -> "img":
        """Full copy of img object"""
        new = {}
        for key in ["img", "ch", "mask"]:
            attr = self.__getattribute__(key)
            if attr is not None:
                new[key] = attr.copy()
            else:
                new[key] = None
        return img(img_arr=new["img"], channels=new["ch"], mask=new["mask"])

    def __getitem__(self, channels):
        """Slice img object with channel name(s)"""
        if isinstance(channels, int):  # force channels into list if single integer
            channels = [channels]
        if isinstance(channels, str):  # force channels into int if single string
            channels = [self.ch.index(channels)]
        if checktype(channels):  # force channels into list of int if list of strings
            channels = [self.ch.index(x) for x in channels]

        if channels is None:  # if no channels are given, use all of them
            channels = [x for x in range(self.n_ch)]

        return self.img[:, :, channels]

    @classmethod
    def from_tiffs(cls, tiffdir, channels, common_strings=None, mask=None):
        """
        Initialize img class from `.tif` files

        Parameters
        ----------
        tiffdir : str
            Path to directory containing `.tif` files for a multiplexed image
        channels : list of str
            List of channels present in `.tif` file names (case-sensitive)
            corresponding to img.shape[2] e.g. `("ACTG1","BCATENIN","DAPI",...)`
        common_strings : str, list of str, or `None`, optional (default=None)
            Strings to look for in all `.tif` files in `tiffdir` corresponding to
            `channels` e.g. `("WD86055_", "_region_001.tif")` for files named
            "WD86055_[MARKERNAME]_region_001.tif". If `None`, assume that only 1 image
            for each marker in `channels` is present in `tiffdir`.
        mask : str, optional (default=None)
            Name of mask defining pixels containing tissue in the image, present in
            `.tif` file names (case-sensitive) e.g. "_01_TISSUE_MASK.tif"

        Returns
        -------
        `img` object
        """
        if common_strings is not None:
            # coerce single string to list
            if isinstance(common_strings, str):
                common_strings = [common_strings]
        A = []  # list for dumping numpy arrays
        for channel in channels:
            if common_strings is None:
                # find file matching all common_strings and channel name
                f = [f for f in os.listdir(tiffdir) if channel in f]
            else:
                # find file matching all common_strings and channel name
                f = [
                    f
                    for f in os.listdir(tiffdir)
                    if all(x in f for x in common_strings + [channel])
                ]
            # assertions so we only get one file per channel
            assert len(f) != 0, "No file found with channel {}".format(channel)
            assert (
                len(f) == 1
            ), "More than one match found for file with channel {}".format(channel)
            f = os.path.join(tiffdir, f[0])  # get full path to file for reading
            print("Reading marker {} from {}".format(channel, f))
            tmp = imread(f)  # read in .tif file
            A.append(tmp)  # append numpy array to list
        A_arr = np.dstack(
            A
        )  # stack numpy arrays in new dimension (third dim is channel)
        print("Final image array of shape: {}".format(A_arr.shape))
        # read in tissue mask if available
        if mask is not None:
            f = [f for f in os.listdir(tiffdir) if mask in f]
            # assertions so we only get one mask file
            assert len(f) != 0, "No tissue mask file found"
            assert len(f) == 1, "More than one match found for tissue mask file"
            f = os.path.join(tiffdir, f[0])  # get full path to file for reading
            print("Reading tissue mask from {}".format(f))
            A_mask = imread(f)  # read in .tif file
            assert (
                A_mask.shape == A_arr.shape[:2]
            ), "Mask (shape: {}) is not the same shape as marker images (shape: {})".format(
                A_mask.shape, A_arr.shape[:2]
            )
            print("Final mask array of shape: {}".format(A_mask.shape))
        else:
            A_mask = None
        # generate img object
        return cls(img_arr=A_arr, channels=channels, mask=A_mask)

    @classmethod
    def from_npz(cls, file):
        """
        Initialize img class from `.npz` file

        Parameters
        ----------
        file : str
            Path to `.npz` file containing saved img object and metadata

        Returns
        -------
        `img` object
        """
        print("Loading img object from {}...".format(file))
        tmp = np.load(file)  # load from .npz compressed file
        assert (
            "img" in tmp.files
        ), "Unexpected files in .npz: {}, expected ['img','mask','ch'].".format(
            tmp.files
        )
        A_mask = tmp["mask"] if "mask" in tmp.files else None
        A_ch = list(tmp["ch"]) if "ch" in tmp.files else None
        # generate img object
        return cls(img_arr=tmp["img"], channels=A_ch, mask=A_mask)

    def to_npz(self, file):
        """
        Save img object to compressed `.npz` file

        Parameters
        ----------
        file : str
            Path to `.npz` file in which to save img object and metadata

        Returns
        -------
        Writes object to `file`
        """
        print("Saving img object to {}...".format(file))
        if self.mask is None:
            np.savez_compressed(file, img=self.img, ch=self.ch)
        else:
            np.savez_compressed(file, img=self.img, ch=self.ch, mask=self.mask)

    def clip(self, **kwargs):
        """
        Clips outlier values and rescales intensities

        Parameters
        ----------
        **kwargs
            Keyword args to pass to `clip_values()` function

        Returns
        -------
        Clips outlier values from `self.img`
        """
        self.img = clip_values(self.img, **kwargs)

    def scale(self, **kwargs):
        """
        Scales intensities to [0.0, 1.0]

        Parameters
        ----------
        **kwargs
            Keyword args to pass to `scale_rgb()` function

        Returns
        -------
        Scales intensities of `self.img`
        """
        self.img = scale_rgb(self.img, **kwargs)

    def equalize_hist(self, **kwargs):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE)

        Parameters
        ----------
        **kwargs
            Keyword args to pass to `CLAHE()` function

        Returns
        -------
        `self.img` is updated with exposure-adjusted values
        """
        self.img = CLAHE(self.img, **kwargs)

    def blurring(self, filter_name="gaussian", sigma=2, **kwargs):
        """
        Aplying a filter on the images

        Parameters
        ----------
        filter : str
            str to define which type of filter to apply

        sigma : int
            parameter controlling extent of smoothening
        """
        if filter_name == "gaussian":
            print("Applying gaussian filter")
            self.img = filters.gaussian(
                self.img,
                sigma=sigma,
                channel_axis=2,
                **kwargs,
            )
        elif filter_name == "median":
            print("Applying median filter")
            if isinstance(sigma, float):
                sigma = int(sigma)
            for i in range(self.img.shape[2]):
                image_array = self.img[:, :, i]
                image_array_blurred = filters.median(
                    image_array,
                    np.ones(sigma, sigma),
                )
                self.img[:, :, i] = image_array_blurred
        elif filter_name == "bilateral":
            print("Applying biltaral filter")
            self.img = denoise_bilateral(
                self.img, sigma_spatial=sigma, channel_axis=2, **kwargs
            )
        else:
            raise Exception(
                "filter name should be either gaussian, median or bilateral"
            )

    def log_normalize(self, pseudoval=1, mean=None, mask=True):
        """
        Log-normalizes values for each marker with `log10(arr/arr.mean() + pseudoval)`

        Parameters
        ----------
        pseudoval : float
            Value to add to image values prior to log-transforming to avoid issues
            with zeros
        mask : bool, optional (default=True)
            Use tissue mask to determine marker mean factor for normalization. Default
            `True`.

        Returns
        -------
        Log-normalizes values in each channel of `self.img`
        """
        if mean is not None:
            if mask:
                assert self.mask is not None, "No tissue mask available"
                for i in range(self.img.shape[2]):
                    fact = mean[i]
                    self.img[:, :, i] = np.log10(self.img[:, :, i] / fact + pseudoval)
            else:
                print("WARNING: Performing normalization without a tissue mask.")
                for i in range(self.img.shape[2]):
                    fact = mean[i]
                    self.img[:, :, i] = np.log10(self.img[:, :, i] / fact + pseudoval)
        else:
            print("mean calculated to perform log normalization")
            if mask:
                assert self.mask is not None, "No tissue mask available"
                for i in range(self.img.shape[2]):
                    fact = self.img[:, :, i].mean()
                    self.img[:, :, i] = np.log10(self.img[:, :, i] / fact + pseudoval)
            else:
                print("WARNING: Performing normalization without a tissue mask.")
                for i in range(self.img.shape[2]):
                    fact = self.img[:, :, i].mean()
                    self.img[:, :, i] = np.log10(self.img[:, :, i] / fact + pseudoval)

    def subsample_pixels(self, features, fract=0.2, random_state=16):
        """
        Sub-samples fraction of pixels from the image randomly for each channel

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        fract : float, optional (default=0.2)
            Fraction of cluster data from each image to randomly select
            for model building

        Returns
        -------
        tmp : np.array
            Clustering data from `image`

        """
        if isinstance(features, int):  # force features into list if single integer
            features = [features]
        if isinstance(features, str):  # force features into int if single string
            features = [self.ch.index(features)]
        if checktype(features):  # force features into list of int if list of strings
            features = [self.ch.index(x) for x in features]
        if features is None:  # if no features are given, use all of them
            features = [x for x in range(self.n_ch)]
        # subsample data for given image
        np.random.seed(random_state)
        tmp = []
        for i in range(self.img.shape[2]):
            tmp.append(self.img[:, :, i][self.mask != 0])
        tmp = np.column_stack(tmp)
        # select cluster data
        i = np.random.choice(tmp.shape[0], int(tmp.shape[0] * fract))
        tmp = tmp[np.ix_(i, features)]
        return tmp

    def downsample(self, fact, func=np.mean):
        """
        Downsamples image by applying `func` to `fact` pixels in both directions from
        each pixel

        Parameters
        ----------
        fact : int
            Number of pixels in each direction (x & y) to downsample with
        func : function
            Numpy function to apply to squares of size (fact, fact, :) for downsampling
            (e.g. `np.mean`, `np.max`, `np.sum`)

        Returns
        -------
        self.img and self.mask are downsampled accordingly in place
        """
        # downsample mask if mask available
        if self.mask is not None:
            self.mask = block_reduce(
                self.mask, block_size=(fact, fact), func=func, cval=0
            )
        # downsample image
        self.img = block_reduce(self.img, block_size=(fact, fact, 1), func=func, cval=0)

    def calculate_non_zero_mean(self):
        """
        Calculate mean estimator for the given image array avoiding mask pixels or
        pixels with value 0

        Parameters
        ----------

        Returns
        -------
        mean_estimator : list of float
            List of mean estimator (mean*pixel) values for each channel
        pixels : int
            pixel count for the image excluding masked pixels
        """
        image = self.img
        pixels = np.count_nonzero(image != 0)
        mean_estimator = []
        for i in range(image.shape[2]):
            ar = image[:, :, i]
            mean = ar[ar != 0].mean()
            mean_estimator.append(mean * pixels)
        return mean_estimator, pixels

    def create_tissue_mask(self, features=None, fract=0.2, sigma = 2):
        """
        Create tissue mask

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        fract : float, optional (default=0.2)
            Fraction of cluster data from each image to randomly select
            for model building

        Returns
        -------
        a numpy array as tissue mask set to self.mask
        """
        if isinstance(features, int):  # force features into list if single integer
            features = [features]
        if isinstance(features, str):  # force features into int if single string
            features = [self.ch.index(features)]
        if checktype(features):  # force features into list of int if list of strings
            features = [self.ch.index(x) for x in features]
        if features is None:  # if no features are given, use all of them
            features = [x for x in range(self.n_ch)]
        # create a copy of the image
        image_cp = self.copy()
        # create a temporary tissue mask that covers no region 
        w, h, d = image_cp.img[:, :, features].shape
        image_cp.mask = np.ones((w, h))
        # log normalization on image
        image_cp.log_normalize()
        # apply gaussian filter
        image_cp.img = filters.gaussian(image_cp.img, sigma=sigma, channel_axis=2)
        # subsample data to build kmeans model
        subsampled_data = image_cp.subsample_pixels(features, fract=fract)
        cluster_data = np.row_stack(subsampled_data)
        # reshape image for prediction
        tmp = image_cp.img[:, :, features]
        image_ar_reshape = tmp.reshape((w * h, d))
        # build kmeans model with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=18).fit(cluster_data)
        labels = kmeans.predict(image_ar_reshape).astype(float)
        tID = labels.reshape((w, h))
        # check if the background is labelled as 0 or 1
        scores = kmeans.cluster_centers_
        mean = scores.mean()
        std = scores.std()
        z_scores = (scores - mean) / std
        if z_scores[0].mean() > 0:
            where_0 = np.where(tID == 0.0)
            tID[where_0] = 0.5
            where_1 = np.where(tID == 1.0)
            tID[where_1] = 0.0
            where_05 = np.where(tID == 0.5)
            tID[where_05] = 1.0
        self.mask = tID

    def show(
        self,
        channels=None,
        RGB=False,
        cbar=False,
        mask_out=True,
        ncols=4,
        figsize=(7, 7),
        save_to=None,
        **kwargs,
    ):
        """
        Plot image

        Parameters
        ----------
        channels : tuple of int or None, optional (default=`None`)
            List of channels by index or name to show
        RGB : bool
            Treat 3- or 4-dimensional array as RGB image. If `False`, plot channels
            individually.
        cbar : bool
            Show colorbar for scale of image intensities if plotting individual
            channels.
        mask_out : bool, optional (default=`True`)
            Mask out non-tissue pixels prior to showing
        ncols : int
            Number of columns for gridspec if plotting individual channels.
        figsize : tuple of float
            Size in inches of output figure.
        save_to : str or None
            Path to image file to save results. If `None`, show figure.
        **kwargs
            Arguments to pass to `plt.imshow()` function.

        Returns
        -------
        Matplotlib object (if plotting one feature or RGB) or gridspec object (for
        multiple features). Saves plot to file if `save_to` is not `None`.
        """
        # if only one feature (2D), plot it quickly
        if self.img.ndim == 2:
            fig = plt.figure(figsize=figsize)
            if self.mask is not None and mask_out:
                im_tmp = self.img.copy()  # make copy for masking
                im_tmp[:, :][self.mask == 0] = np.nan  # area outside mask to NaN
                plt.imshow(im_tmp, **kwargs)
            else:
                plt.imshow(self.img, **kwargs)
            plt.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            if cbar:
                plt.colorbar(shrink=0.8)
            plt.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=800
                )
            return fig
        # if image has multiple channels, plot them in gridspec
        if isinstance(channels, int):  # force channels into list if single integer
            channels = [channels]
        if isinstance(channels, str):  # force channels into int if single string
            channels = [self.ch.index(channels)]
        if checktype(channels):  # force channels into list of int if list of strings
            channels = [self.ch.index(x) for x in channels]
        if channels is None:  # if no channels are given, use all of them
            channels = [x for x in range(self.n_ch)]
        assert (
            len(channels) <= self.n_ch
        ), "Too many channels given: image has {}, expected {}".format(
            self.n_ch, len(channels)
        )
        if RGB:
            # if third dim has 3 or 4 features, treat as RGB and plot it quickly
            assert (self.img.ndim == 3) & (
                len(channels) == 3
            ), "Need 3 dimensions and 3 given channels for an RGB image; shape = {}; channels given = {}".format(
                self.img.shape, len(channels)
            )
            fig = plt.figure(figsize=figsize)
            # rearrange channels to specified order
            im_tmp = np.dstack(
                [
                    self.img[:, :, channels[0]],
                    self.img[:, :, channels[1]],
                    self.img[:, :, channels[2]],
                ]
            )
            if self.mask is not None and mask_out:
                for i in [0, 1, 2]:  # for 3-channel image
                    im_tmp[:, :, i][self.mask == 0] = np.nan  # area outside mask NaN
            plt.imshow(im_tmp, **kwargs)
            # add legend for channel IDs
            custom_lines = [
                Line2D([0], [0], color=(1, 0, 0), lw=5),
                Line2D([0], [0], color=(0, 1, 0), lw=5),
                Line2D([0], [0], color=(0, 0, 1), lw=5),
            ]
            plt.legend(custom_lines, [self.ch[x] for x in channels], fontsize="medium")
            plt.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            plt.tight_layout()
            if save_to:
                plt.savefig(
                    fname=save_to, transparent=True, bbox_inches="tight", dpi=300
                )
            return fig
        # calculate gridspec dimensions
        if len(channels) <= ncols:
            n_rows, n_cols = 1, len(channels)
        else:
            n_rows, n_cols = ceil(len(channels) / ncols), ncols
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        i = 0
        for channel in channels:
            ax = plt.subplot(gs[i])
            if self.mask is not None and mask_out:
                im_tmp = self.img[:, :, channel].copy()  # make copy for masking
                im_tmp[self.mask == 0] = np.nan  # area outside mask NaN
                im = ax.imshow(im_tmp, **kwargs)
            else:
                im = ax.imshow(self.img[:, :, channel], **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=self.ch[channel],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            if cbar:
                _ = plt.colorbar(im, shrink=0.8)
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig

    def plot_image_histogram(self, channels=None, ncols=4, save_to=None, **kwargs):

        """
        Plot image histogram

        Parameters
        ----------
        channels : tuple of int or None, optional (default=`None`)
            List of channels by index or name to show
        ncols : int
            Number sof columns for gridspec if plotting individual channels.
        save_to : str or None
            Path to image file to save results. If `None`, show figure.
        **kwargs
            Arguments to pass to `plt.imshow()` function.

        Returns
        -------
        Gridspec object (for multiple features). Saves plot to file if `save_to` is
        not `None`.
        """
        # calculate gridspec dimensions
        if len(channels) <= ncols:
            n_rows, n_cols = 1, len(channels)
        else:
            n_rows, n_cols = ceil(len(channels) / ncols), ncols
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        i = 0
        for channel in channels:
            ax = plt.subplot(gs[i])
            data = self[channel].copy()
            ax.hist(data.ravel(), bins=100, **kwargs)
            ax.set_title(channel, fontweight="bold", fontsize=16)
            i = i + 1
        fig.tight_layout()
        plt.show()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return gs
