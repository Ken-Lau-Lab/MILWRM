# -*- coding: utf-8 -*-
"""
Classes for assigning tissue domain IDs to multiplex immunofluorescence (MxIF) or 10X 
Visium spatial transcriptomic (ST) and histological imaging data
"""
import os
from tkinter import E
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import umap

sns.set_style("white")

from math import ceil
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from .MxIF import checktype, img
from .ST import assemble_pita
from .ST import blur_features_st


def kMeansRes(scaled_data, k, alpha_k=0.02, random_state=18):
    """
    Calculates inertia value for a given k value by fitting k-means model to scaled data

    Adapted from
    https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

    Parameters
    ----------
    scaled_data: np.array
        Scaled data. Rows are samples and columns are features for clustering
    k: int
        Current k for applying KMeans
    alpha_k: float
        Manually tuned factor that gives penalty to the number of clusters

    Returns
    -------
    scaled_inertia: float
        Scaled inertia value for current k
    """
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def chooseBestKforKMeansParallel(scaled_data, k_range, n_jobs=-1, **kwargs):
    """
    Determines optimal k value by fitting k-means models to scaled data and minimizing
    scaled inertia

    Adapted from
    https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

    Parameters
    ----------
    scaled_data: np.array
        Scaled data. Rows are samples and columns are features for clustering.
    k_range: list of int
        k range for applying KMeans
    n_jobs : int
        Number of cores to parallelize k-choosing across
    **kwargs
        Arguments to pass to `kMeansRes()` (i.e. `alpha_k`, `random_state`)

    Returns
    -------
    best_k: int
        Chosen value of k out of the given k range. Chosen k is k with the minimum
        scaled inertia value.
    results: pd.DataFrame
        Adjusted inertia value for each k in k_range
    """
    ans = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(kMeansRes)(scaled_data, k, **kwargs) for k in k_range
    )
    ans = list(zip(k_range, ans))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k, results


def prep_data_single_sample_st(
    adata,
    adata_i,
    use_rep,
    features,
    histo,
    fluor_channels,
    spatial_graph_key=None,
    n_rings=1,
):
    """
    Prepare dataframe for tissue-level clustering from a single AnnData sample

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing Visium data
    adata_i : int
        Index of AnnData object for identification within `st_labeler` object
    use_rep : str
        Representation from `adata.obsm` to use as clustering data (e.g. "X_pca")
    features : list of int or None, optional (default=`None`)
        List of features to use from `adata.obsm[use_rep]` (e.g. [0,1,2,3,4] to
        use first 5 principal components when `use_rep`="X_pca"). If `None`, use
        all features from `adata.obsm[use_rep]`
    histo : bool, optional (default `False`)
        Use histology data from Visium anndata object (R,G,B brightfield features)
        in addition to `adata.obsm[use_rep]`? If fluorescent imaging data rather
        than brightfield, use `fluor_channels` argument instead.
    fluor_channels : list of int or None, optional (default `None`)
        Channels from fluorescent image to use for model training (e.g. [1,3] for
        channels 1 and 3 of Visium fluorescent imaging data). If `None`, do not
        use imaging data for training.
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
    pd.DataFrame
        Clustering data from `adata.obsm[use_rep]`
    """
    tmp = pd.DataFrame()
    tmp[[use_rep + "_{}".format(x) for x in features]] = adata.obsm[use_rep][
        :, features
    ]
    if histo:
        assert (
            fluor_channels is None
        ), "If histo is True, fluor_channels must be None. \
            Histology specifies brightfield H&E with three (3) features."
        print("Adding mean RGB histology features for adata #{}".format(adata_i))
        tmp[["R_mean", "G_mean", "B_mean"]] = adata.obsm["image_means"]
    if fluor_channels:
        assert (
            histo is False
        ), "If fluorescence channels are given, histo must be False. \
            Histology specifies brightfield H&E with three (3) features."
        print(
            "Adding mean fluorescent channels {} for adata #{}".format(
                fluor_channels, adata_i
            )
        )
        tmp[["ch_{}_mean".format(x) for x in fluor_channels]] = adata.obsm[
            "image_means"
        ][:, fluor_channels]
    if n_rings > 0:
        # blur the features extracted in tmp
        tmp = blur_features_st(
            adata, tmp, spatial_graph_key=spatial_graph_key, n_rings=n_rings
        )
    return tmp


def prep_data_single_sample_mxif(
    image, use_path, mean, filter_name, sigma, features, fract, path_save
):
    """
    Perform log normalization, and blurring on the given image data

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is
        given as MILWRM.MxIF.img object
    mean : numpy array
        Containing mean for each channel for that batch
    filter_name : str
        Name of the filter to use - gaussian, median or bilateral
    sigma : float, optional
        Standard deviation of Gaussian kernel for blurring
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    fract : float, optional
        Fraction of cluster data from each image to randomly select for model
        building
    path_save : str
        Path to save final preprocessed files
    Returns
    -------
    Subsampled_data : np.array
        np.array containing randomly sampled pixels for that image
    file_save : str
        path to save preprocessed img object
    """
    if use_path == True:  # if images are given as path to the compressed npz file
        if path_save == None:  # check if path to save final processed file is given
            raise Exception(
                "Path to save final preprocessed npz files is requird when given path to image files"
            )
        image_path = image
        image = img.from_npz(image_path + ".npz")
    # batch correction
    image.log_normalize(mean=mean)
    # apply the desired filter
    image.blurring(filter_name=filter_name, sigma=sigma)
    # min max scaling of each channel
    # for i in range(image.img.shape[2]):
    #     img_ar = image.img[:, :, i][image.mask != 0]
    #     img_ar_max = img_ar.max()
    #     img_ar_min = img_ar.min()
    #     # print(img_ar_max, img_ar_min)
    #     image_ar_scaled = (image.img[:, :, i] - img_ar_min) / (img_ar_max - img_ar_min)
    #     image.img[:, :, i] = image_ar_scaled
    # subsample pixels to build the kmeans model
    subsampled_data = image.subsample_pixels(features, fract)
    if use_path == True:
        new_image_path = os.path.join(path_save, "_final_preprocessed_images")
        if not os.path.exists(new_image_path):
            os.mkdir(new_image_path)
        file_name = image_path.split("/")[-1] + "_final_preprocessed"
        file_save = os.path.join(new_image_path, file_name)
        image.to_npz(file_save)
        return subsampled_data, file_save
    return subsampled_data


def add_tissue_ID_single_sample_mxif(image, use_path, features, kmeans, scaler):
    """
    Label pixels in a single MxIF sample with kmeans results

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is
        given as MILWRM.MxIF.img object
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    kmeans : sklearn.kmeans
        Trained k-means model

    Returns
    -------
    tID : np.array
        Image where pixel values are kmeans cluster IDs
    """
    if use_path == True:
        image_path = image + ".npz"
        image = img.from_npz(image_path)
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    if isinstance(features, str):  # force features into int if single string
        features = [image.ch.index(features)]
    if checktype(features):  # force features into list of int if list of strings
        features = [image.ch.index(x) for x in features]
    if features is None:  # if no features are given, use all of them
        features = [x for x in range(image.n_ch)]
    # subset to features used in prep_cluster_data
    tmp = image.img[:, :, features]
    w, h, d = tuple(tmp.shape)
    image_array = tmp.reshape((w * h, d))
    scaled_image_array = scaler.transform(image_array)
    tID = kmeans.predict(scaled_image_array).reshape(w, h)
    tID = tID.astype(float)  # TODO: Figure out dtypes
    tID[image.mask == 0] = np.nan  # set masked-out pixels to NaN
    return tID


def estimate_percentage_variance_mxif(
    image, use_path, scaler, centroids, features, tissue_ID
):
    """
    Estimate percentage variance explained by clustering for an image

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is
        given as MILWRM.MxIF.img object
    scaler : standardscaler() object
        standard scaler used for cluster data normalization
    centroids : np.ndarray
        kmeans cluster centroids
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    tissue_ID : np.ndarray
        numpy array containing kmeans labels on image

    Returns
    -------
    S_square_pct : float
        percentage variance in data explained by the kmeans clustering

    """
    if use_path == True:
        image_path = image + ".npz"
        image = img.from_npz(image_path)
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    if isinstance(features, str):  # force features into int if single string
        features = [image.ch.index(features)]
    if checktype(features):  # force features into list of int if list of strings
        features = [image.ch.index(x) for x in features]
    if features is None:  # if no features are given, use all of them
        features = [x for x in range(image.n_ch)]
    # getting the channels used for MILWRM clustering and scaling the image
    w, h, d = image.img[:, :, features].shape
    img_ar = image.img[:, :, features].reshape((w * h), d)
    scaled_img_ar = scaler.transform(img_ar)
    tissue_ID = tissue_ID.reshape(w * h)
    # init a numpy array of image shape to store the distance from pixels to centroids
    dc = np.zeros(scaled_img_ar.shape)
    for i in range(centroids.shape[0]):
        dc[tissue_ID == i] = ((scaled_img_ar[tissue_ID == i]) - (centroids[i])) ** 2
    # estimating the difference between pixels and the image mean
    dm = ((scaled_img_ar) - (scaled_img_ar.mean(axis=0))) ** 2
    # taking ratio of sum of differences for all points from centroids and data mean
    S_square = np.sum(dc) / np.sum(dm)
    S_square_pct = S_square * 100
    return S_square_pct


def perform_umap(cluster_data, centroids, batch_labels, kmeans_labels, frac):
    """
    Compute umap coordinates for the given cluster_data

    Parameters
    ----------
    cluster_data : np.ndarray
        containing data used to build kmeans model
    centroids : np.ndarray
        kmeans cluster centroids
    batch_labels : list
        list containing batch label for each datapoint
    kmeans_label : list
        list containing tissue domain labels for each datapoint
    frac : None or float
        if None entire cluster_data is used to compute umap if float
        that fraction of data is used to compute the umap

    Returns
    -------
    umap_centroid_data : pd.DataFrame
        combined dataframe with cluster_data used for computation
        of Umap, centroids, batch_labels and kmeans_labels
    standard_embedding : pd.DataFrame
        containing umap coordinates
    """
    df = pd.DataFrame(cluster_data, batch_labels)
    df["Kmeans_labels"] = kmeans_labels
    # if cluster_data is too big randomly subsample a fraction of it otherwise use the
    # entire data
    if frac:
        umap_data = pd.DataFrame()
        for i in np.unique(batch_labels):
            umap_data = pd.concat([umap_data, df.loc[i].sample(frac=frac)])
    else:
        umap_data = df
    # append the centroids to the dataframe with a different index and kmeans labels
    centroids = pd.DataFrame(
        centroids, index=[umap_data.index[-1] + 1] * len(centroids)
    )
    centroids["Kmeans_labels"] = [kmeans_labels.max() + 1] * len(centroids)
    umap_centroid_data = pd.concat([umap_data, centroids])
    # compute umap
    neighbours = int(len(umap_centroid_data) ** 0.5)
    mapper = umap.UMAP(random_state=42, n_neighbors=neighbours).fit(
        umap_centroid_data.loc[:, umap_centroid_data.columns != "Kmeans_labels"]
    )
    standard_embedding = mapper.transform(
        umap_centroid_data.loc[:, umap_centroid_data.columns != "Kmeans_labels"]
    )
    return umap_centroid_data, standard_embedding


def estimate_confidence_score_mxif(
    image, use_path, scaler, centroids, features, tissue_ID
):
    """
    Estimate confidence score for the assigned tissue_IDs in MxIF slide by
    taking difference between distance to the second closest centroid and the
    assigned centroid divided by distance to the second closest centroid.

    Parameters
    ----------
    image : MILWRM.MxIF.img or str
        np.array containing MxIF data or path to the compressed npz file
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is
        given as MILWRM.MxIF.img object
    scaler : standardscaler() object
        standard scaler used for cluster data normalization
    centroids : np.ndarray
        kmeans cluster centroids
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    tissue_ID : np.ndarray
        numpy array containing kmeans labels on image

    Returns
    -------
    Conf_ID : np.ndarray
        overall confidence score for each pixel's cluster assignment
    mean_conf_score : dict
        mean confidence score for each tissue domain (keys for the dictionary)
    """
    if use_path == True:
        image_path = image + ".npz"
        image = img.from_npz(image_path)
    if isinstance(features, int):  # force features into list if single integer
        features = [features]
    if isinstance(features, str):  # force features into int if single string
        features = [image.ch.index(features)]
    if checktype(features):  # force features into list of int if list of strings
        features = [image.ch.index(x) for x in features]
    if features is None:  # if no features are given, use all of them
        features = [x for x in range(image.n_ch)]
    w, h, d = image.img[:, :, features].shape
    img_ar = image.img[:, :, features].reshape((w * h), d)
    scaled_img_ar = scaler.transform(img_ar)
    img_sc = scaled_img_ar.reshape((w, h, d))
    # initializing an empty numpy array to store distance to each centroid along axis = 2
    dist_ar = np.zeros((w, h, len(centroids)))
    for i, centroid in enumerate(centroids):
        dist = ((img_sc) - (centroid)) ** 2
        dist_cp = np.sum(dist, axis=2)
        dist_ar[:, :, i] = dist_ar[:, :, i] + dist_cp
    # sorting the numpy array according to distance from centroids
    new_dist_ar = np.sort(dist_ar, axis=2)
    # estimating new confidence score
    cID = ((new_dist_ar[:, :, 1]) - (new_dist_ar[:, :, 0])) / new_dist_ar[:, :, 1]
    cID[image.mask == 0] = np.nan
    # estimating average confidence score in that image
    mean_conf_score = {}
    for i in range(len(centroids)):
        mean_conf_score[i] = np.mean(cID[tissue_ID == i])
    return cID, mean_conf_score


def estimate_mse_mxif(images, use_path, tissue_IDs, scaler, centroids, features, k):
    """
    Estimate mean square error for each tissue domain for each MxIF images

    Parameters
    ----------
    images : list
        list of MILWRM.MxIF.img objects or path to images (str)
    use_path : Boolean
        True if image is given as a path to the compressed npz file, False if image is
        given as MILWRM.MxIF.img object
    tissue_IDs : list
        list of predicted tissue_ID for each image
    scaler : standardscaler() object
        standard scaler used for cluster data normalization
    centroids : np.ndarray
        kmeans cluster centroids
    features : list of int or str
        Indices or names of MxIF channels to use for tissue labeling
    k : int
        number of tissue domains

    Returns
    -------
    mse_id : dict
        containing mean square error for each tissue for each mxif slide
    """
    mse_temp = {}
    for image_index, image in enumerate(images):
        if use_path == True:
            image_path = image + ".npz"
            image = img.from_npz(image_path)
        if isinstance(features, int):  # force features into list if single integer
            features = [features]
        if isinstance(features, str):  # force features into int if single string
            features = [image.ch.index(features)]
        if checktype(features):  # force features into list of int if list of strings
            features = [image.ch.index(x) for x in features]
        if features is None:  # if no features are given, use all of them
            features = [x for x in range(image.n_ch)]
        # getting the channels used for MILWRM clustering and scaling the image
        img_ar = image.img[:, :, features]
        w, h, d = img_ar.shape
        scaled_img_ar = scaler.transform(img_ar.reshape((w * h, d)))
        scaled_img_ar = scaled_img_ar.reshape((w, h, d))
        ar = tissue_IDs[image_index]
        mse = {}
        for i in range(k):
            x = (
                (scaled_img_ar[ar == i]) - (centroids[i])
            ) ** 2  # estimating mse for each tissue domain for that image
            if len(x) == 0:
                mse[i] = np.zeros((centroids.shape[1]))
            else:
                mse[i] = x.mean(axis=0)
        mse_temp[image_index] = mse
    mse_id = {}  # reorganizing within a new dictionary with keys as tissue domains
    for i in range(k):
        mse_l = []
        for image_index, image in enumerate(images):
            mse_l.append(mse_temp[image_index][i])
            mse_id[i] = mse_l
    return mse_id


def estimate_percentage_variance_st(sub_cluster_data, adata, centroids):
    """
    Estimate percentage variance explained by clustering for a visium slide

    Parameters
    ----------
    sub_cluster_data : np.ndarray
        np.ndarray containing data for that visium slide used for kmeans
    adata : anndata.AnnData
        AnnData object containing Visium data
    centroids : np.ndarray
        kmeans cluster centroids

    Returns
    -------
    S_square_pct : float
        percentage variance in data explained by the kmeans clustering

    """
    dc = []
    df = pd.DataFrame(adata.obs["tissue_ID"])
    ids = pd.unique(df["tissue_ID"])
    df["index"] = list(range(adata.n_obs))
    for i in ids:
        # estimating euclidean distance from the data point to closest centroid
        diff = (
            (sub_cluster_data[df[df["tissue_ID"] == i]["index"]]) - (centroids[i])
        ) ** 2
        dc.append(diff)
    dc = np.row_stack(dc)
    # estimating euclidean distance from each data point to the mean of the data
    dm = (sub_cluster_data - sub_cluster_data.mean(axis=0)) ** 2
    # getting sum across features
    # taking ratio of sum of distances for all data points from centroids and data mean
    S = np.sum(dc) / np.sum(dm)
    S_square_pct = S * 100
    return S_square_pct


def estimate_confidence_score_st(sub_cluster_data, adata, centroids):
    """
    Estimate confidence score for the assigned tissue_IDs in a visium slide by
    taking difference between distance to the second closest centroid and the
    assigned centroid divided by distance to the second closest centroid.

    Parameters
    ----------
    sub_cluster_data : np.ndarray
        np.ndarray containing data for that visium slide used for kmeans
    adata : anndata.AnnData
        AnnData object containing Visium data
    centroids : np.ndarray
        kmeans cluster centroids

    Returns
    -------
    Confidence_score added to adata.obs
    mean_conf_score : dict
        mean confidence score for each tissue domain (keys for the dictionary)
    """
    # initializing zeros array to store distances
    dist_mx = np.zeros((sub_cluster_data.shape[0], len(centroids)))
    # calculating distance to each centroid
    for i, centroid in enumerate(centroids):
        dist_cp = ((sub_cluster_data) - (centroid)) ** 2
        dist = np.sum(dist_cp, axis=1)
        dist_mx[:, i] = dist_mx[:, i] + dist
    # sorting distances according to distance from centroids
    new_dist_ar = np.sort(dist_mx, axis=1)
    # using assigned and second closest centroid to estimate confidence score
    cID = ((new_dist_ar[:, 1]) - (new_dist_ar[:, 0])) / new_dist_ar[:, 1]
    adata.obs["confidence_score"] = cID
    score_df = pd.DataFrame(cID, columns=["score"])
    score_df["tissue_ID"] = adata.obs["tissue_ID"].values
    mean_conf_score = {}
    for i in range(len(centroids)):
        if (adata.obs["tissue_ID"] == i).any():
            mean_conf_score[i] = score_df[score_df["tissue_ID"] == i]["score"].mean()
        else:
            mean_conf_score[i] = np.nan
    return mean_conf_score


def estimate_mse_st(cluster_data, adatas, centroids, k):
    """
    Estimate mean square error for each tissue domain for each visium slide

    Parameters
    ----------
    cluster_data : np.ndarray
        np.ndarray containing cluster_data used for kmeans
    adatas :  list
        list of anndata.AnnData objects for visium slides
    centroids : np.ndarray
        kmeans cluster centroids
    k : int
        number of tissue domains

    Returns
    -------
    mse_id : dict
        containing mean square error for each tissue for each visium slide
    """
    mse_id = {}
    for i in range(k):
        i_slice = 0
        j_slice = 0
        diff = []
        for adata in adatas:
            j_slice = j_slice + adata.n_obs
            df = pd.DataFrame(adata.obs["tissue_ID"])
            df["index"] = list(range(adata.n_obs))
            data = cluster_data[
                i_slice:j_slice
            ]  # slicing cluster data for sub_cluster_data for that visium slide
            x = (
                (data[df[df["tissue_ID"] == i]["index"]]) - (centroids[i])
            ) ** 2  # difference between each data point and centroids
            if len(x) == 0:
                diff.append(np.zeros((centroids.shape[1])))
            else:
                mse = x.mean(axis=0)  # mean of all the differences
                # diff.append(mse.mean(axis = 0))
                diff.append(mse)
            i_slice = adata.n_obs
        mse_id[i] = diff
    return mse_id


class tissue_labeler:
    """
    Master tissue domain labeling class
    """

    def __init__(self):
        """
        Initialize tissue labeler parent class
        """
        self.cluster_data = None  # start out with no data to cluster on
        self.k = None  # start out with no k value

    def find_optimal_k(self, plot_out=False, alpha=0.05, random_state=18, n_jobs=-1):
        """
        Uses scaled inertia to decide on k clusters for clustering in the
        corresponding `anndata` objects

        Parameters
        ----------
        plot_out : boolean, optional (default=FALSE)
            Determines if scaled inertia graph should be output
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        random_state : int, optional (default=18)
            Seed for k-means clustering models
        n_jobs : int
            Number of cores to parallelize k-choosing across

        Returns
        -------
        Does not return anything. `self.k` contains integer value for number of
        clusters. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            raise Exception("No cluster data found. Run prep_cluster_data() first.")
        self.random_state = random_state

        k_range = range(2, 21)  # choose k range
        # compute scaled inertia
        best_k, results = chooseBestKforKMeansParallel(
            self.cluster_data,
            k_range,
            n_jobs=n_jobs,
            random_state=random_state,
            alpha_k=alpha,
        )
        if plot_out:
            # plot the results
            plt.figure(figsize=(7, 4))
            plt.plot(results, "o")
            plt.title("Adjusted Inertia for each K")
            plt.xlabel("K")
            plt.ylabel("Adjusted Inertia")
            plt.xticks(range(2, 21, 1))
            plt.show()
        # save optimal k to object
        print("The optimal number of clusters is {}".format(best_k))
        self.k = best_k

    def find_tissue_regions(self, k=None, random_state=18):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue domains to define
        random_state : int, optional (default=18)
            Seed for k-means clustering model.

        Returns
        -------
        Does not return anything. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is None:
            raise Exception("No cluster data found. Run prep_cluster_data() first.")
        if k is None and self.k is None:
            raise Exception(
                "No k found or provided. Run find_optimal_k() first or pass a k value."
            )
        if k is not None:
            print("Overriding optimal k value with k={}.".format(k))
            self.k = k
        # save the hyperparams as object attributes
        self.random_state = random_state
        print("Performing k-means clustering with {} target clusters".format(self.k))
        self.kmeans = KMeans(n_clusters=self.k, random_state=random_state).fit(
            self.cluster_data
        )

    def plot_feature_proportions(self, labels=None, figsize=(10, 7), save_to=None):
        """
        Plots contributions of each training feature to k-means cluster centers as
        percentages of total

        Parameters
        ----------
        labels : list of str, optional (default=`None`)
            Labels corresponding to each MILWRM training feature. If `None`, features
            will be numbered 0 through p.
        figsize : tuple of float, optional (default=(10,7))
            Size of matplotlib figure
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `plt.figure` if `save_to` is `None`, else saves plot to file
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        if "st_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = [self.rep + "_" + str(x) for x in self.features]
                if self.histo:
                    labels = labels + ["R", "G", "B"]
                if self.fluor_channels is not None:
                    labels = labels + ["ch_" + str(x) for x in self.fluor_channels]
        elif "mxif_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = self.model_features
        # create pandas df and calculate percentages of total
        ctr_df = pd.DataFrame(self.kmeans.cluster_centers_, columns=labels)
        totals = ctr_df.sum(axis=1)
        ctr_df_prop = ctr_df.div(totals, axis=0).multiply(100)
        # make plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ctr_df_prop.plot.bar(stacked=True, ax=ax, width=0.85)
        for p in ax.patches:
            ax.annotate(
                "{} %".format(str(np.round(p.get_height(), 2))),
                (p.get_x() + 0.05, p.get_y() + (p.get_height() * 0.4)),
                fontsize=10,
            )
        plt.ylim([0, 100])
        plt.xlabel("tissue_ID")
        plt.xticks(rotation=0)
        plt.ylabel("% K-Means Loading")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Feature")
        plt.tight_layout()
        if save_to is not None:
            print("Saving feature proportions to {}".format(save_to))
            plt.savefig(save_to)
        else:
            return fig

    def plot_feature_loadings(
        self,
        ncols=None,
        nfeatures=None,
        labels=None,
        titles=None,
        figsize=(5, 5),
        save_to=None,
    ):
        """
        Plots contributions of each training feature to k-means cluster centers

        Parameters
        ----------
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        nfeatures : int, optional (default=`None`)
            Number of top-loaded features to show for each tissue domain
        labels : list of str, optional (default=`None`)
            Labels corresponding to each MILWRM training feature. If `None`, features
            will be numbered 0 through p.
        titles : list of str, optional (default=`None`)
            Titles of plots corresponding to each MILWRM domain. If `None`, titles
            will be numbers 0 through k.
        figsize : tuple of float, optional (default=(5,5))
            Size of matplotlib figure
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `gridspec.GridSpec` if `save_to` is `None`, else saves plot to file
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        if "st_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = [self.rep + "_" + str(x) for x in self.features]
                if self.histo:
                    labels = labels + ["R", "G", "B"]
                if self.fluor_channels is not None:
                    labels = labels + ["ch_" + str(x) for x in self.fluor_channels]
        elif "mxif_labeler" in str(self.__class__):
            if labels is not None:
                assert len(labels) == len(
                    self.features
                ), "'labels' must be the same length as self.features."
            else:
                labels = self.model_features
        if titles is None:
            titles = [
                "tissue_ID " + str(x)
                for x in range(self.kmeans.cluster_centers_.shape[0])
            ]
        if nfeatures is None:
            nfeatures = len(labels)
        scores = self.kmeans.cluster_centers_.copy()
        n_panels = len(titles)
        if ncols is None:
            ncols = len(titles)
        if n_panels <= ncols:
            n_rows, n_cols = 1, n_panels
        else:
            n_rows, n_cols = ceil(n_panels / ncols), ncols
        fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        left, bottom = 0.1 / n_cols, 0.1 / n_rows
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=n_cols,
            wspace=0.1,
            left=left,
            bottom=bottom,
            right=1 - (n_cols - 1) * left - 0.01 / n_cols,
            top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
        )
        for iscore, score in enumerate(scores):
            plt.subplot(gs[iscore])
            indices = np.argsort(score)[::-1][: nfeatures + 1]
            for ig, g in enumerate(indices[::-1]):
                plt.text(
                    x=score[g],
                    y=ig,
                    s=labels[g],
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize="medium",
                    fontstyle="italic",
                )
            plt.title(titles[iscore], fontsize="x-large")
            plt.ylim(-0.9, ig + 0.9)
            score_min, score_max = np.min(score[indices]), np.max(score[indices])
            plt.xlim(
                (0.95 if score_min > 0 else 1.05) * score_min,
                (1.05 if score_max > 0 else 0.95) * score_max,
            )
            plt.xticks(rotation=45)
            plt.tick_params(labelsize="medium")
            plt.tick_params(
                axis="y",  # changes apply to the y-axis
                which="both",  # both major and minor ticks are affected
                left=False,
                right=False,
                labelleft=False,
            )
            plt.grid(False)
        gs.tight_layout(fig)
        if save_to is not None:
            print("Saving feature loadings to {}".format(save_to))
            plt.savefig(save_to)
        else:
            return gs


class st_labeler(tissue_labeler):
    """
    Tissue domain labeling class for spatial transcriptomics (ST) data
    """

    def __init__(self, adatas):
        """
        Initialize ST tissue labeler class

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Single anndata object or list of objects to label consensus tissue domains

        Returns
        -------
        Does not return anything. `self.adatas` attribute is updated,
        `self.cluster_data` attribute is initiated as `None`.
        """
        tissue_labeler.__init__(self)  # initialize parent class
        if not isinstance(adatas, list):  # force single anndata object to list
            adatas = [adatas]
        print("Initiating ST labeler with {} anndata objects".format(len(adatas)))
        self.adatas = adatas
        self.raw = adatas.copy()

    def prep_cluster_data(
        self,
        use_rep,
        features=None,
        n_rings=1,
        histo=False,
        fluor_channels=None,
        spatial_graph_key=None,
        n_jobs=-1,
    ):
        """
        Prepare master dataframe for tissue-level clustering

        Parameters
        ----------
        use_rep : str
            Representation from `adata.obsm` to use as clustering data (e.g. "X_pca")
        features : list of int or None, optional (default=`None`)
            List of features to use from `adata.obsm[use_rep]` (e.g. [0,1,2,3,4] to
            use first 5 principal components when `use_rep`="X_pca"). If `None`, use
            all features from `adata.obsm[use_rep]`
        n_rings : int, optional (default=1)
            Number of hexagonal rings around each spatial transcriptomics spot to blur
            features by for capturing regional information. Assumes 10X Genomics Visium
            platform.
        histo : bool, optional (default `False`)
            Use histology data from Visium anndata object (R,G,B brightfield features)
            in addition to `adata.obsm[use_rep]`? If fluorescent imaging data rather
            than brightfield, use `fluor_channels` argument instead.
        fluor_channels : list of int or None, optional (default `None`)
            Channels from fluorescent image to use for model training (e.g. [1,3] for
            channels 1 and 3 of Visium fluorescent imaging data). If `None`, do not
            use imaging data for training.
        spatial_graph_key : str, optional (default=`None`)
            Key in `adata.obsp` containing spatial graph connectivities (i.e.
            `"spatial_connectivities"`). If `None`, compute new spatial graph using
            `n_rings` in `squidpy`.
        n_jobs : int, optional (default=-1)
            Number of cores to parallelize over. Default all available cores.

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "blur_*" features
        to `.obs` if `n_rings > 0`.
        `self.cluster_data` becomes master `np.array` for cluster training.
        Parameters are also captured as attributes for posterity.
        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        if features is None:
            self.features = [x for x in range(self.adatas[0].obsm[use_rep].shape[1])]
        else:
            self.features = features
        # save the hyperparams as object attributes
        self.rep = use_rep
        self.histo = histo
        self.fluor_channels = fluor_channels
        self.n_rings = n_rings
        # collect clustering data from self.adatas in parallel
        print(
            "Collecting and blurring {} features from .obsm[{}]...".format(
                len(self.features),
                use_rep,
            )
        )
        cluster_data = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(prep_data_single_sample_st)(
                adata,
                adata_i,
                use_rep,
                self.features,
                histo,
                fluor_channels,
                spatial_graph_key,
                n_rings,
            )
            for adata_i, adata in enumerate(self.adatas)
        )
        batch_labels = [
            [x] * len(cluster_data[x]) for x in range(len(cluster_data))
        ]  # batch labels for umap
        self.merged_batch_labels = list(itertools.chain(*batch_labels))
        # concatenate blurred features into cluster_data df for cluster training
        subsampled_data = pd.concat(cluster_data)
        # perform z-scaling on final cluster data
        scaler = StandardScaler()
        self.scaler = scaler.fit(subsampled_data)
        scaled_data = scaler.transform(subsampled_data)
        self.cluster_data = scaled_data
        print("Collected clustering data of shape: {}".format(self.cluster_data.shape))

    def label_tissue_regions(
        self, k=None, alpha=0.05, plot_out=True, random_state=18, n_jobs=-1
    ):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        `anndata` objects.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        plot_out : boolean, optional (default=True)
            Determines if scaled inertia plot should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model.
        n_jobs : int
            Number of cores to parallelize k-choosing across

        Returns
        -------
        Does not return anything. `self.adatas` are updated, adding "tissue_ID" field
        to `.obs`. `self.kmeans` contains trained `sklearn` clustering model.
        Parameters are also captured as attributes for posterity.
        """
        # find optimal k with parent class
        if k is None:
            print("Determining optimal cluster number k via scaled inertia")
            self.find_optimal_k(
                plot_out=plot_out, alpha=alpha, random_state=random_state, n_jobs=n_jobs
            )
        # call k-means model from parent class
        self.find_tissue_regions(k=k, random_state=random_state)
        # loop through anndata object and add tissue labels to adata.obs dataframe
        start = 0
        print("Adding tissue_ID label to anndata objects")
        for i in range(len(self.adatas)):
            IDs = self.kmeans.labels_
            self.adatas[i].obs["tissue_ID"] = IDs[start : start + self.adatas[i].n_obs]
            self.adatas[i].obs["tissue_ID"] = (
                self.adatas[i].obs["tissue_ID"].astype("category")
            )
            self.adatas[i].obs["tissue_ID"] = (
                self.adatas[i].obs["tissue_ID"].cat.set_categories(np.unique(IDs))
            )
            start += self.adatas[i].n_obs

    def confidence_score(self):
        """
        estimate confidence score for each visium slide

        Parameters
        ----------

        Returns
        -------
        self.adatas[i].obs.confidence_IDs and self.confidence_score_df are added
        containing confidence score for each tissue domain assignment and mean confidence
        score for each tissue domain within each visium slide
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        i_slice = 0
        j_slice = 0
        confidence_score_df = pd.DataFrame()
        adatas = self.adatas
        cluster_data = self.cluster_data
        centroids = self.kmeans.cluster_centers_
        for i, adata in enumerate(adatas):
            j_slice = j_slice + adata.n_obs
            data = cluster_data[i_slice:j_slice]
            scores_dict = estimate_confidence_score_st(data, adata, centroids)
            df = pd.DataFrame(scores_dict.values(), columns=[i])
            confidence_score_df = pd.concat([confidence_score_df, df], axis=1)
            i_slice = i_slice + adata.n_obs
        self.confidence_score_df = confidence_score_df

    def plot_gene_loadings(
        self,
        PC_loadings,
        n_genes=10,
        ncols=None,
        titles=None,
        save_to=None,
    ):
        """
        Plot MILWRM loadings in gene space specifically for MILWRM done with PCs

        Parameters
        ----------
        PC_loadings : numpy.ndarray
            numpy.ndarray containing PC loadings shape format (genes, components)
        n_genes : int, optional (default=10)
            number of genes to plot
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        titles : list of str, optional (default=`None`)
            Titles of plots corresponding to each MILWRM domain. If `None`, titles
            will be numbers 0 through k.
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        Matplotlib object and PC loadings in gene space set as self.gene_loadings_df
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        assert (
            PC_loadings.shape[0] == self.adatas[0].n_vars
        ), f"loadings matrix does not, \
        contain enough genes, there should be {self.adatas[0].n_vars} genes"
        assert (
            PC_loadings.shape[1] >= self.kmeans.cluster_centers_.shape[1]
        ), f"loadings matrix \
        does not contain enough components, there should be atleast {self.adatas[0].n_vars} components"
        if titles is None:
            titles = ["tissue_ID " + str(x) for x in range(self.k)]
        centroids = self.kmeans.cluster_centers_
        temp = PC_loadings.T
        loadings = temp[range(self.kmeans.cluster_centers_.shape[1])]
        gene_loadings = np.matmul(centroids, loadings)
        gene_loadings_df = pd.DataFrame(gene_loadings)
        gene_loadings_df = gene_loadings_df.T
        gene_loadings_df["genes"] = self.adatas[0].var_names
        self.gene_loadings_df = gene_loadings_df
        n_panels = self.k
        if ncols is None:
            ncols = self.k
        if n_panels <= ncols:
            n_rows, n_cols = 1, n_panels
        else:
            n_rows, n_cols = ceil(n_panels / ncols), ncols
        fig = plt.figure(figsize=((ncols * n_cols, ncols * n_rows)))
        left, bottom = 0.1 / n_cols, 0.1 / n_rows
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=n_cols,
            left=left,
            bottom=bottom,
            right=1 - (n_cols - 1) * left - 0.01 / n_cols,
            top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
        )
        for i in range(self.k):
            df = (
                gene_loadings_df[[i, "genes"]]
                .sort_values(i, axis=0, ascending=False)[:n_genes]
                .reset_index(drop=True)
            )
            plt.subplot(gs[i])
            df_rev = df.sort_values(i).reset_index(drop=True)
            for j, score in enumerate((df_rev[i])):
                plt.text(
                    x=score,
                    y=j + 0.1,
                    s=df_rev.loc[j, "genes"],
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize="medium",
                    fontstyle="italic",
                )
                plt.ylim([0, j + 1])
                plt.xlim([0, df.max().values[0] + 0.1])
                plt.tick_params(
                    axis="y",  # changes apply to the y-axis
                    which="both",  # both major and minor ticks are affected
                    left=False,
                    right=False,
                    labelleft=False,
                )
                plt.title(titles[i])
        gs.tight_layout(fig)
        if save_to is not None:
            print("Saving feature loadings to {}".format(save_to))
            plt.savefig(save_to)
        else:
            return gs

    def plot_percentage_variance_explained(
        self, fig_size=(5, 5), R_square=False, save_to=None
    ):
        """
        plot percentage variance_explained or not explained by clustering

        Parameters
        ----------
        figsize : tuple of float, optional (default=(5,5))
            Size of matplotlib figure
        R_square : Boolean
            Decides if R_square is plotted or S_square
        save_to : str or None
            Path to image file to save results. If `None`, show figure.

        Returns
        -------
        Matplotlib object
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        centroids = self.kmeans.cluster_centers_
        adatas = self.adatas
        cluster_data = self.cluster_data
        S_squre_for_each_st = []
        R_squre_for_each_st = []
        i_slice = 0
        j_slice = 0
        for adata in adatas:
            j_slice = j_slice + adata.n_obs
            sub_cluster_data = cluster_data[i_slice:j_slice]
            S_square = estimate_percentage_variance_st(
                sub_cluster_data, adata, centroids
            )
            S_squre_for_each_st.append(S_square)
            R_squre_for_each_st.append(100 - S_square)
            i_slice = i_slice + adata.n_obs

        if R_square:
            fig = plt.figure(figsize=fig_size)
            plt.scatter(
                range(len(R_squre_for_each_st)), R_squre_for_each_st, color="black"
            )
            plt.xlabel("images")
            plt.ylabel("percentage variance explained by Kmeans")
            plt.ylim((0, 100))
            plt.axhline(
                y=np.mean(R_squre_for_each_st),
                linestyle="dashed",
                linewidth=1,
                color="black",
            )

        else:
            fig = plt.figure(figsize=fig_size)
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(
                range(len(S_squre_for_each_st)), S_squre_for_each_st, color="black"
            )
            plt.xlabel("images")
            plt.ylabel("percentage variance explained by Kmeans")
            plt.ylim((0, 100))
            plt.axhline(
                y=np.mean(S_squre_for_each_st),
                linestyle="dashed",
                linewidth=1,
                color="black",
            )

        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig

    def plot_mse_st(
        self,
        figsize=(5, 5),
        ncols=None,
        labels=None,
        titles=None,
        loc="lower right",
        bbox_coordinates=(0, 0, 1.5, 1.5),
        save_to=None,
    ):
        """
        estimate mean square error within each tissue domain

        Parameters
        ----------
        fig_size : Tuple
            size for the bar plot
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        labels : list of str, optional (default=`None`)
            Labels corresponding to each image in legend. If `None`, numeric index is
            used for each imaage
        titles : list of str, optional (default=`None`)
            Titles of plots corresponding to each MILWRM domain. If `None`, titles
            will be numbers 0 through k.
        loc : str, optional (default = 'lower right')
            str for legend position
        bbox_coordinates : Tuple, optional (default = (0,0,1.5,1.5))
            coordinates for the legend box
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        Matplotlib object
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        cluster_data = self.cluster_data
        adatas = self.adatas
        k = self.k
        features = self.features
        centroids = self.kmeans.cluster_centers_
        mse_id = estimate_mse_st(cluster_data, adatas, centroids, k)
        colors = plt.cm.tab20(np.linspace(0, 1, len(adatas)))
        if titles is None:
            titles = ["tissue_domain " + str(x) for x in range(self.k)]
        if labels is None:
            labels = range(len(adatas))
        n_panels = len(mse_id.keys())
        if ncols is None:
            ncols = len(titles)
        if n_panels <= ncols:
            n_rows, n_cols = 1, n_panels
        else:
            n_rows, n_cols = ceil(n_panels / ncols), ncols
        fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        left, bottom = 0.1 / n_cols, 0.1 / n_rows
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=n_cols,
            left=left,
            bottom=bottom,
            right=1 - (n_cols - 1) * left - 0.01 / n_cols,
            top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
        )
        for i in mse_id.keys():
            plt.subplot(gs[i])
            df = pd.DataFrame.from_dict(mse_id[i])
            plt.boxplot(df, positions=features, showfliers=False)
            for col in df:
                for k in range(len(df[col])):
                    dots = plt.scatter(
                        col,
                        df[col][k],
                        s=k + 1,
                        color=colors[k],
                        label=labels[k] if col == 0 else "",
                    )
                    offsets = dots.get_offsets()
                    jittered_offsets = offsets
                    # only jitter in the x-direction
                    jittered_offsets[:, 0] += np.random.uniform(
                        -0.3, 0.3, offsets.shape[0]
                    )
                    dots.set_offsets(jittered_offsets)
            plt.xlabel("PCs")
            plt.ylabel("mean square error")
            plt.title(titles[i])
        plt.legend(loc=loc, bbox_to_anchor=bbox_coordinates)
        gs.tight_layout(fig)
        if save_to:
            plt.savefig(fname=save_to, transparent=True, dpi=300)
        return fig

    def plot_tissue_ID_proportions_st(
        self,
        tID_labels=None,
        slide_labels=None,
        figsize=(5, 5),
        cmap="tab20",
        save_to=None,
    ):
        """
        Plot proportion of each tissue domain within each slide

        Parameters
        ----------
        tID_labels : list of str, optional (default=`None`)
            List of labels corresponding to MILWRM tissue domains for plotting legend
        slide_labels : list of str, optional (default=`None`)
            List of labels for each slide batch for labeling x-axis
        figsize : tuple of float, optional (default=(5,5))
            Size of matplotlib figure
        cmap : str, optional (default = `"tab20"`)
            Colormap from matplotlib
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `gridspec.GridSpec` if `save_to` is `None`, else saves plot to file
        """
        df_count = pd.DataFrame()
        for adata in self.adatas:
            df = adata.obs["tissue_ID"].value_counts(normalize=True, sort=False)
            df_count = pd.concat([df_count, df], axis=1)
        df_count = df_count.T.reset_index(drop=True)
        if tID_labels:
            assert (
                len(tID_labels) == df_count.shape[1]
            ), "Length of given tissue domain labels does not match number of tissue domains!"
            df_count.columns = tID_labels
        if slide_labels:
            assert (
                len(slide_labels) == df_count.shape[0]
            ), "Length of given slide labels does not match number of slides!"
            df_count.index = slide_labels
        ax = df_count.plot.bar(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc="best", bbox_to_anchor=(1, 1))
        ax.set_xlabel("slides")
        ax.set_ylabel("tissue domain proportion")
        ax.set_ylim((0, 1))
        plt.tight_layout()
        if save_to is not None:
            ax.figure.savefig(save_to)
        else:
            return ax

    def show_feature_overlay(
        self,
        adata_index,
        pita,
        features=None,
        histo=None,
        cmap="tab20",
        label="feature",
        ncols=4,
        save_to=None,
        **kwargs,
    ):
        """
        Plot tissue_ID with individual pita features as alpha values to distinguish
        expression in identified tissue domains

        Parameters
        ----------
        adata_index : int
            Index of adata from `self.adatas` to plot overlays for (e.g. 0 for first
            adata object)
        pita : np.array
            Image of desired expression in pixel space from `.assemble_pita()`
        features : list of int, optional (default=`None`)
            List of features by index to show in plot. If `None`, use all features.
        histo : np.array or `None`, optional (default=`None`)
            Histology image to show along with pita in gridspec. If `None`, ignore.
        cmap : str, optional (default="tab20")
            Matplotlib colormap to use for plotting tissue domains
        label : str
            What to title each panel of the gridspec (i.e. "PC" or "usage") or each
            channel in RGB image. Can also pass list of names e.g. ["NeuN","GFAP",
            "DAPI"] corresponding to channels.
        ncols : int
            Number of columns for gridspec
        save_to : str or None
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
        # create tissue_ID pita for plotting
        tIDs = assemble_pita(
            self.adatas[adata_index],
            features="tissue_ID",
            use_rep="obs",
            plot_out=False,
            verbose=False,
        )
        # if pita has multiple features, plot them in gridspec
        if isinstance(features, int):  # force features into list if single integer
            features = [features]
        # if no features are given, use all of them
        elif features is None:
            features = [x + 1 for x in range(pita.shape[2])]
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
        # min-max scale each feature in pita to convert to interpretable alpha values
        mms = MinMaxScaler()
        if pita.ndim == 3:
            pita_tmp = mms.fit_transform(
                pita.reshape((pita.shape[0] * pita.shape[1], pita.shape[2]))
            )
        elif pita.ndim == 2:
            pita_tmp = mms.fit_transform(
                pita.reshape((pita.shape[0] * pita.shape[1], 1))
            )
        # reshape back to original
        pita = pita_tmp.reshape(pita.shape)
        # figure out labels for gridspec plots
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
            # determine where the histo image is in anndata
            assert (
                histo
                in self.adatas[adata_index]
                .uns["spatial"][
                    list(self.adatas[adata_index].uns["spatial"].keys())[0]
                ]["images"]
                .keys()
            ), "Must provide one of {} for histo".format(
                self.adatas[adata_index]
                .uns["spatial"][
                    list(self.adatas[adata_index].uns["spatial"].keys())[0]
                ]["images"]
                .keys()
            )
            histo = self.adatas[adata_index].uns["spatial"][
                list(self.adatas[adata_index].uns["spatial"].keys())[0]
            ]["images"][histo]
            if len(features) + 2 <= ncols:
                n_rows, n_cols = 1, len(features) + 2
            else:
                n_rows, n_cols = ceil((len(features) + 2) / ncols), ncols
            labels = ["Histology", "tissue_ID"] + labels  # append to front of labels
        else:
            if len(features) + 1 <= ncols:
                n_rows, n_cols = 1, len(features) + 1
            else:
                n_rows, n_cols = ceil(len(features) + 1 / ncols), ncols
            labels = ["tissue_ID"] + labels  # append to front of labels
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # add plots to axes
        i = 0
        if histo is not None:
            # add histology plot to first axes
            ax = plt.subplot(gs[i])
            im = ax.imshow(histo, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=labels[i],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            i = i + 1
        # plot tissue_ID first with colorbar
        ax = plt.subplot(gs[i])
        im = ax.imshow(tIDs, cmap=cmap, **kwargs)
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        ax.set_title(
            label=labels[i],
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        # colorbar scale for tissue_IDs
        _ = plt.colorbar(im, shrink=0.7)
        i = i + 1
        for feature in features:
            ax = plt.subplot(gs[i])
            im = ax.imshow(tIDs, alpha=pita[:, :, feature - 1], cmap=cmap, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=labels[i],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig


class mxif_labeler(tissue_labeler):
    """
    Tissue domain labeling class for multiplex immunofluorescence (MxIF) data
    """

    def __init__(self, image_df):
        """
        Initialize MxIF tissue labeler class

        Parameters
        ----------
        image_df : pd.DataFrame object
            Containing MILWRM.MxIF.img objects or str path to compressed npz files,
            batch names, mean estimator and pixel count for each image in the
            following column order ['Img', 'batch_names', 'mean estimators', 'pixels']

        Returns
        -------
        Does not return anything. `self.images` attribute is updated,
        `self.cluster_data` attribute is initiated as `None`.
        """
        tissue_labeler.__init__(self)  # initialize parent class
        # validate the format of the image_df dataframe
        if np.all(
            image_df.columns == ["Img", "batch_names", "mean estimators", "pixels"]
        ):
            self.image_df = image_df
        else:
            raise Exception(
                "Image_df must be given with these columns in this format ['Img', 'batch_names', 'mean estimators', 'pixels']"
            )
        if self.image_df["Img"].apply(isinstance, args=[img]).all():
            self.use_paths = False
        elif self.image_df["Img"].apply(isinstance, args=[str]).all():
            self.use_paths = True
        else:
            raise Exception(
                "Img column in the dataframe should be either str for paths to the files or mxif.img object"
            )

    def prep_cluster_data(
        self, features, filter_name="gaussian", sigma=2, fract=0.2, path_save=None
    ):
        """
        Prepare master array for tissue level clustering

        Parameters
        ----------
        features : list of int or str
            Indices or names of MxIF channels to use for tissue labeling
        filter_name : str
            Name of the filter to use - gaussian, median or bilateral
        sigma : float, optional (default=2)
            Standard deviation of Gaussian kernel for blurring
        fract : float, optional (default=0.2)
            Fraction of cluster data from each image to randomly select for model
            building
        path_save : str (default = None)
            Path to save final preprocessed files, if self.use_path is True
            default path_save will raise Exception

        Returns
        -------
        Does not return anything. `self.images` are normalized, blurred and scaled
        according to user parameters. `self.cluster_data` becomes master `np.array`
        for cluster training. Parameters are also captured as attributes for posterity.

        """
        if self.cluster_data is not None:
            print("WARNING: overwriting existing cluster data")
            self.cluster_data = None
        # save the hyperparams as object attributes
        self.model_features = features
        use_path = self.use_paths
        # calculate the batch wise means
        mean_for_each_batch = {}
        for batch in self.image_df["batch_names"].unique():
            list_mean_estimators = list(
                self.image_df[self.image_df["batch_names"] == batch]["mean estimators"]
            )
            mean_estimator_batch = sum(map(np.array, list_mean_estimators))
            pixels = sum(self.image_df[self.image_df["batch_names"] == batch]["pixels"])
            mean_for_each_batch[batch] = mean_estimator_batch / pixels
        # log_normalize, apply blurring filter, minmax scale each channel and subsample
        subsampled_data = []
        path_to_blurred_npz = []
        for image, batch in zip(self.image_df["Img"], self.image_df["batch_names"]):
            tmp = prep_data_single_sample_mxif(
                image,
                use_path=use_path,
                mean=mean_for_each_batch[batch],
                filter_name=filter_name,
                sigma=sigma,
                features=self.model_features,
                fract=fract,
                path_save=path_save,
            )
            if self.use_paths == True:
                subsampled_data.append(tmp[0])
                path_to_blurred_npz.append(tmp[1])
            else:
                subsampled_data.append(tmp)
        batch_labels = [
            [x] * len(subsampled_data[x]) for x in range(len(subsampled_data))
        ]  # batch labels for umap
        self.merged_batch_labels = list(itertools.chain(*batch_labels))
        if self.use_paths == True:
            self.image_df["Img"] = path_to_blurred_npz
        cluster_data = np.row_stack(subsampled_data)
        # perform z-score normalization on cluster_Data
        scaler = StandardScaler()
        self.scaler = scaler.fit(cluster_data)
        scaled_data = scaler.transform(cluster_data)
        self.cluster_data = scaled_data

    def label_tissue_regions(
        self, k=None, alpha=0.05, plot_out=True, random_state=18, n_jobs=-1
    ):
        """
        Perform tissue-level clustering and label pixels in the corresponding
        images.

        Parameters
        ----------
        k : int, optional (default=None)
            Number of tissue regions to define
        alpha: float
            Manually tuned factor on [0.0, 1.0] that penalizes the number of clusters
        plot_out : boolean, optional (default=True)
            Determines if scaled inertia plot should be output
        random_state : int, optional (default=18)
            Seed for k-means clustering model
        n_jobs : int
            Number of cores to parallelize k-choosing and tissue domain assignment across.
            Default all available cores.

        Returns
        -------
        Does not return anything. `self.tissue_ID` is added, containing image with
        final tissue region IDs. `self.kmeans` contains trained `sklearn` clustering
        model. Parameters are also captured as attributes for posterity.
        """
        # save the hyperparams as object attributes
        use_path = self.use_paths
        # find optimal k with parent class
        if k is None:
            print("Determining optimal cluster number k via scaled inertia")
            self.find_optimal_k(
                alpha=alpha,
                plot_out=plot_out,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        # call k-means model from parent class
        self.find_tissue_regions(k=k, random_state=random_state)
        # loop through image objects and create tissue label images
        print("Creating tissue_ID images for image objects...")
        self.tissue_IDs = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(add_tissue_ID_single_sample_mxif)(
                image, use_path, self.model_features, self.kmeans, self.scaler
            )
            for image in self.image_df["Img"]
        )

    def plot_percentage_variance_explained(
        self, fig_size=(5, 5), R_square=False, save_to=None
    ):
        """
        plot percentage variance_explained or not explained by clustering

        Parameters
        ----------
        fig_size : Tuple
            size for the bar plot
        R_square : Boolean
            Decides if R_square is plotted or S_square
        save_to : str or None
            Path to image file to save results. If `None`, show figure.

        Returns
        -------
        Matplotlib object
        """
        scaler = self.scaler
        centroids = self.kmeans.cluster_centers_
        features = self.model_features
        use_path = self.use_paths
        S_squre_for_each_image = []
        R_squre_for_each_image = []
        for image, tissue_ID in zip(self.image_df["Img"], self.tissue_IDs):
            S_square = estimate_percentage_variance_mxif(
                image, use_path, scaler, centroids, features, tissue_ID
            )
            S_squre_for_each_image.append(S_square)
            R_squre_for_each_image.append(100 - S_square)

        if R_square == True:
            fig = plt.figure(figsize=fig_size)
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(
                range(len(R_squre_for_each_image)),
                R_squre_for_each_image,
                color="black",
            )
            plt.xlabel("images")
            plt.ylabel("percentage variance not explained by Kmeans")
            plt.ylim((0, 100))
            plt.axhline(
                y=np.mean(R_squre_for_each_image),
                linestyle="dashed",
                linewidth=1,
                color="black",
            )

        else:
            fig = plt.figure(figsize=fig_size)
            plt.scatter(
                range(len(S_squre_for_each_image)),
                S_squre_for_each_image,
                color="black",
            )
            plt.xlabel("images")
            plt.ylabel("percentage variance explained by Kmeans")
            plt.ylim((0, 100))
            plt.axhline(
                y=np.mean(S_squre_for_each_image),
                linestyle="dashed",
                linewidth=1,
                color="black",
            )

        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig

    def confidence_score_images(self):
        """
        estimate confidence score for each image

        Parameters
        ----------

        Returns
        -------
        self.confidence_IDs and self.confidence_score_df is added containing
        confidence score for each tissue domain assignment and mean confidence score for
        each tissue domain within each image
        """
        scaler = self.scaler
        centroids = self.kmeans.cluster_centers_
        features = self.model_features
        tissue_IDs = self.tissue_IDs
        use_path = self.use_paths
        # confidence score estimation for each image
        confidence_IDs = []
        confidence_score_df = pd.DataFrame()
        for i, image in enumerate(self.image_df["Img"]):
            cID, scores_dict = estimate_confidence_score_mxif(
                image, use_path, scaler, centroids, features, tissue_IDs[i]
            )
            confidence_IDs.append(cID)
            df = pd.DataFrame(scores_dict.values(), columns=[i])
            confidence_score_df = pd.concat(
                [confidence_score_df, df.T], ignore_index=True
            )
        # adding confidence_IDs and confidence_score_df to tissue labeller object
        self.confidence_IDs = confidence_IDs
        self.confidence_score_df = confidence_score_df

    def plot_mse_mxif(
        self,
        figsize=(5, 5),
        ncols=None,
        labels=None,
        legend_cols=2,
        titles=None,
        loc="lower right",
        bbox_coordinates=(0, 0, 1.5, 1.5),
        save_to=None,
    ):
        """
        estimate mean square error within each tissue domain

        Parameters
        ----------
        fig_size : Tuple
            size for the bar plot
        ncols : int, optional (default=`None`)
            Number of columns for gridspec. If `None`, uses number of tissue domains k.
        labels : list of str, optional (default=`None`)
            Labels corresponding to each image in legend. If `None`, numeric index is
            used for each imaage
        legend_cols : int, optional (default = `2`)
            n_cols for legend
        titles : list of str, optional (default=`None`)
            Titles of plots corresponding to each MILWRM domain. If `None`, titles
            will be numbers 0 through k.
        loc : str, optional (default = 'lower right')
            str for legend position
        bbox_coordinates : Tuple, optional (default = (0,0,1.5,1.5))
            coordinates for the legend box
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        Matplotlib object
        """
        assert (
            self.kmeans is not None
        ), "No cluster results found. Run \
        label_tissue_regions() first."
        images = self.image_df["Img"]
        use_path = self.use_paths
        scaler = self.scaler
        centroids = self.kmeans.cluster_centers_
        features = self.model_features
        k = self.k
        features = self.model_features
        tissue_IDs = self.tissue_IDs
        mse_id = estimate_mse_mxif(
            images, use_path, tissue_IDs, scaler, centroids, features, k
        )
        if labels is None:
            labels = range(len(images))
        if titles is None:
            titles = ["tissue_ID " + str(x) for x in range(self.k)]
        n_panels = len(mse_id.keys())
        if ncols is None:
            ncols = len(titles)
        if n_panels <= ncols:
            n_rows, n_cols = 1, n_panels
        else:
            n_rows, n_cols = ceil(n_panels / ncols), ncols
        colors = plt.cm.tab20(np.linspace(0, 1, len(images)))
        fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
        left, bottom = 0.1 / n_cols, 0.1 / n_rows
        gs = gridspec.GridSpec(
            nrows=n_rows,
            ncols=n_cols,
            left=left,
            bottom=bottom,
            right=1 - (n_cols - 1) * left - 0.01 / n_cols,
            top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
        )
        for i in mse_id.keys():
            plt.subplot(gs[i])
            df = pd.DataFrame.from_dict(mse_id[i])
            plt.boxplot(df, positions=range(len(features)), showfliers=False)
            plt.xticks(
                ticks=range(len(features)),
                labels=self.model_features,
                rotation=60,
                fontsize=8,
            )
            for col in df:
                for k in range(len(images)):
                    dots = plt.scatter(
                        col,
                        df[col][k],
                        s=k + 1,
                        color=colors[k],
                        label=labels[k] if col == 0 else "",
                    )
                    offsets = dots.get_offsets()
                    jittered_offsets = offsets
                    # only jitter in the x-direction
                    jittered_offsets[:, 0] += np.random.uniform(
                        -0.3, 0.3, offsets.shape[0]
                    )
                    dots.set_offsets(jittered_offsets)
            plt.xlabel("marker")
            plt.ylabel("mean square error")
            plt.title(titles[i])
        plt.legend(loc=loc, bbox_to_anchor=bbox_coordinates, ncol=legend_cols)
        gs.tight_layout(fig)
        if save_to:
            plt.savefig(fname=save_to, transparent=True, dpi=300)
        return fig

    def plot_tissue_ID_proportions_mxif(
        self,
        tID_labels=None,
        slide_labels=None,
        figsize=(5, 5),
        cmap="tab20",
        save_to=None,
    ):
        """
        Plot proportion of each tissue domain within each slide

        Parameters
        ----------
        tID_labels : list of str, optional (default=`None`)
            List of labels corresponding to MILWRM tissue domains for plotting legend
        slide_labels : list of str, optional (default=`None`)
            List of labels for each slide batch for labeling x-axis
        figsize : tuple of float, optional (default=(5,5))
            Size of matplotlib figure
        cmap : str, optional (default = `"tab20"`)
        save_to : str, optional (default=`None`)
            Path to image file to save plot

        Returns
        -------
        `gridspec.GridSpec` if `save_to` is `None`, else saves plot to file
        """
        df_count = pd.DataFrame()
        for i in range(len(self.tissue_IDs)):
            unique, counts = np.unique(self.tissue_IDs[i], return_counts=True)
            dict_ = dict(zip(unique, counts))
            n_counts = []
            for k in range(self.k):
                if k not in dict_.keys():
                    n_counts.append(0)
                else:
                    n_counts.append(dict_[k])
            df = pd.DataFrame(n_counts, columns=[i])
            df_count = pd.concat([df_count, df], axis=1)
        df_count = df_count / df_count.sum()
        if tID_labels:
            assert (
                len(tID_labels) == df_count.shape[1]
            ), "Length of given tissue domain labels does not match number of tissue domains!"
            df_count.columns = tID_labels
        if slide_labels:
            assert (
                len(slide_labels) == df_count.shape[0]
            ), "Length of given slide labels does not match number of slides!"
            df_count.index = slide_labels
        self.tissue_ID_proportion = df_count
        ax = df_count.T.plot.bar(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc="best", bbox_to_anchor=(1, 1))
        ax.set_xlabel("images")
        ax.set_ylabel("tissue domain proportion")
        ax.set_ylim((0, 1))
        plt.tight_layout()
        if save_to is not None:
            ax.figure.savefig(save_to)
        else:
            return ax

    def make_umap(self, frac=None, cmap="tab20", save_to=None, alpha=0.8, dot_size_batch = 0.1):
        """
        plot umap for the cluster data

        Parameters
        ----------
        frac : None or float
            if None entire cluster data is used for the computation of umap
            else that percentage of cluster data is used.
        cmap : str
            str for cmap used for plotting. Default `"tab20"`.
        save_to : str or None
            Path to image file to save results. if `None`, show figure.
        alpha : float
            opaqueness of umap scatter plot (default=`0.8`)
        dot_size_batch = float
            scatter plot dot size (default=`0.1`)

        Returns
        -------
        Matplotlib object
        """
        cluster_data = self.cluster_data
        centroids = self.kmeans.cluster_centers_
        batch_labels = self.merged_batch_labels
        kmeans_labels = self.kmeans.labels_
        k = self.k
        # perform umap on the cluster data
        umap_centroid_data, standard_embedding_1 = perform_umap(
            cluster_data=cluster_data,
            centroids=centroids,
            batch_labels=batch_labels,
            kmeans_labels=kmeans_labels,
            frac=frac,
        )
        # defining a size of datapoints for scatter plot and tick labels
        size = [0.01] * len(umap_centroid_data.index)
        size[-k:] = [10] * k
        ticks = np.unique(np.array(umap_centroid_data["Kmeans_labels"]))
        tick_label = list(np.unique(np.array(umap_centroid_data["Kmeans_labels"])))
        tick_label[-1] = "centroids"
        # plotting a fig with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        # defining color_map
        disc_cmap_1 = plt.cm.get_cmap(
            cmap, len(np.unique(np.array(umap_centroid_data.index)))
        )
        disc_cmap_2 = plt.cm.get_cmap(
            cmap, len(np.unique(np.array(umap_centroid_data["Kmeans_labels"])))
        )
        plot_1 = ax1.scatter(
            standard_embedding_1[:, 0],
            standard_embedding_1[:, 1],
            s=dot_size_batch,
            c=umap_centroid_data.index,
            cmap=disc_cmap_1,
            alpha=alpha,
        )
        ax1.set_title("UMAP with batch labels", fontsize = 24)
        ax1.set_xlabel("UMAP 2")
        ax1.set_ylabel("UMAP 1")
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar_1 = plt.colorbar(plot_1, ax=ax1)

        plot_2 = ax2.scatter(
            standard_embedding_1[:, 0],
            standard_embedding_1[:, 1],
            s=size,
            c=umap_centroid_data["Kmeans_labels"],
            cmap=disc_cmap_2,
            alpha=alpha,
        )
        ax2.set_title("UMAP with tissue domains", fontsize = 24)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("UMAP 2")
        ax2.set_ylabel("UMAP 1")
        cbar_2 = plt.colorbar(plot_2, ax=ax2, ticks=ticks)
        cbar_2.ax.set_yticklabels(tick_label)
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig

    def show_marker_overlay(
        self,
        image_index,
        channels=None,
        cmap="Set1",
        mask_out=True,
        ncols=4,
        save_to=None,
        **kwargs,
    ):
        """
        Plot tissue_ID with individual markers as alpha values to distinguish
        expression in identified tissue domains

        Parameters
        ----------
        image_index : int
            Index of image from `self.images` to plot overlays for (e.g. 0 for first
            image)
        channels : tuple of int or None, optional (default=`None`)
            List of channels by index or name to show
        cmap : str, optional (default="plasma")
            Matplotlib colormap to use for plotting tissue domains
        mask_out : bool, optional (default=`True`)
            Mask out non-tissue pixels prior to showing
        ncols : int
            Number of columns for gridspec if plotting individual channels.
        save_to : str or None
            Path to image file to save results. If `None`, show figure.
        **kwargs
            Arguments to pass to `plt.imshow()` function.

        Returns
        -------
        Matplotlib object (if plotting one feature or RGB) or gridspec object (for
        multiple features). Saves plot to file if `save_to` is not `None`.
        """
        # if image has multiple channels, plot them in gridspec
        if isinstance(channels, int):  # force channels into list if single integer
            channels = [channels]
        if isinstance(channels, str):  # force channels into int if single string
            channels = [self[image_index].ch.index(channels)]
        if checktype(channels):  # force channels into list of int if list of strings
            channels = [self[image_index].ch.index(x) for x in channels]
        if channels is None:  # if no channels are given, use all of them
            channels = [x for x in range(self[image_index].n_ch)]
        assert (
            len(channels) <= self[image_index].n_ch
        ), "Too many channels given: image has {}, expected {}".format(
            self[image_index].n_ch, len(channels)
        )
        # creating a copy of the image
        image_cp = self[image_index].copy()
        # re-scaling to set pixel value range between 0 to 1
        image_cp.scale()
        # defining cmap for discrete color bar
        cmap = plt.cm.get_cmap(cmap, self.k)
        # calculate gridspec dimensions
        if len(channels) + 1 <= ncols:
            n_rows, n_cols = 1, len(channels) + 1
        else:
            n_rows, n_cols = ceil(len(channels) + 1 / ncols), ncols
        fig = plt.figure(figsize=(ncols * n_cols, ncols * n_rows))
        # arrange axes as subplots
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
        # plot tissue_ID first with colorbar
        ax = plt.subplot(gs[0])
        im = ax.imshow(self.tissue_IDs[image_index], cmap=cmap, **kwargs)
        ax.set_title(
            label="tissue_ID",
            loc="left",
            fontweight="bold",
            fontsize=16,
        )
        ax.tick_params(labelbottom=False, labelleft=False)
        sns.despine(bottom=True, left=True)
        # colorbar scale for tissue_IDs
        _ = plt.colorbar(im, ticks=range(self.k), shrink=0.7)
        # add plots to axes
        i = 1
        for channel in channels:
            ax = plt.subplot(gs[i])
            # make copy for alpha
            im_tmp = image_cp.img[:, :, channel].copy()
            if self[image_index].mask is not None and mask_out:
                # area outside mask NaN
                self.tissue_IDs[image_index][self[image_index].mask == 0] = np.nan
                im = ax.imshow(
                    self.tissue_IDs[image_index], cmap=cmap, alpha=im_tmp, **kwargs
                )
            else:
                ax.imshow(self.tissue_IDs[image_index], alpha=im_tmp, **kwargs)
            ax.tick_params(labelbottom=False, labelleft=False)
            sns.despine(bottom=True, left=True)
            ax.set_title(
                label=self[image_index].ch[channel],
                loc="left",
                fontweight="bold",
                fontsize=16,
            )
            i = i + 1
        fig.tight_layout()
        if save_to:
            plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=300)
        return fig
