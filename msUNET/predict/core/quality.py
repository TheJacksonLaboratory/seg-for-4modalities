# author: Zachary Frohock
'''
Functions pertaining to post-inference quality checks
quality_check() is the primary function handling qc
'''

import sys
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import ImageDraw, Image
from scipy.spatial import ConvexHull
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    '''
    Context that suppresses output to stdout
    '''
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def low_snr_check(current_slice):
    '''
    Function that calculated signal to noise ratio. Compares circle of pixels
    in image center to those on top left corner.
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of source data stack
    Output
    ----------
    snr: float
        Signal to noise ratio
    '''
    i, j = np.indices(current_slice.shape)
    center_mean = np.array(
        current_slice[((i-(current_slice.shape[0]//2))**2 < 9)
                      & ((j-(current_slice.shape[1]//2))**2 < 9)]).mean()
    k, l = 3, 3
    edge_mean = current_slice[max(0, k-3):k+3, max(0, l-3):l+3].mean()
    if edge_mean == 0:
        edge_mean = 0.001

    return center_mean/edge_mean


def mask_area_check(current_slice, current_slice_source):
    '''
    Function that calculates the fraction of image canvas occupied by pixels
    categorized as brain and fraction of image canvas occupied by pixels with
    intensity greater than the mean value of all pixels in source data
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of mask stack
    current_slice_source: array like (_, _)
        Single slice of source data stack
    Outputs
    ----------
    mask_ratio: Float
        Fraction of image canvas occupied by pixels categorized as brain
    source_data_ratio: Float
        Fraction of image canvas occupied by pixels with intensity greater
        than the mean over current slice
    '''
    total_pixels = current_slice.size
    mask_pixels = (np.asarray(current_slice) > 0).sum()
    source_data_pixels = (np.asarray(current_slice_source)
                          > current_slice_source.mean()).sum()

    mask_ratio = mask_pixels/total_pixels
    source_data_ratio = source_data_pixels/total_pixels

    return mask_ratio, source_data_ratio


def convex_hull_image(data):
    '''
    Function that calculates and draws the convex hull for a 2D binary image
    Parameters
    ----------
    data: array like (_, _)
        2D image data, from a slice
    Outputs
    ----------
    mask: array like (_, _)
        Binary mask corresponding to convex hull region
    '''
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v, 0], region[v, 1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def solidity_check(current_slice):
    '''
    Function that calculates solidity of mask
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of mask stack
    Output
    ----------
    slice_solidity: Float
        Solidity of mask in current slice
    '''
    current_slice_convex_hull = convex_hull_image(current_slice)

    mask_pixels = (current_slice == 1).sum()
    convex_hull_pixels = (current_slice_convex_hull == 1).sum()

    slice_solidity = mask_pixels/convex_hull_pixels

    return slice_solidity


def otsu_snr_check(current_slice):
    '''
    Function that calculates signal to noise ratio using an otsu binarization
    scheme for determining foreground and background
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of source data stack
    Outputs
    ----------
    otsu_snr: float
        signal to noise ratio of foreground and background
    masked_array_std: float
        Standard deviation of foreground intensity
    masked_background_std: float
        Standard deviation of background intensity
    otsu_fraction: float
        Fraction of image canvas occupied by foreground
    masked_array_mean: float
        Mean intensity value of foreground
    masked_background_mean: float
        Mean intensity value of background
    '''
    current_image = sitk.GetImageFromArray(current_slice)

    otsu_image_filter = sitk.OtsuThresholdImageFilter()
    otsu_image_filter.SetInsideValue(0)
    otsu_image_filter.SetOutsideValue(1)
    otsu_mask = otsu_image_filter.Execute(current_image)
    otsu_array = sitk.GetArrayFromImage(otsu_mask)
    otsu_background = np.where((otsu_array == 0) | (otsu_array == 1),
                               otsu_array ^ 1, otsu_array)

    masked_array = np.nan_to_num(np.multiply(current_slice, otsu_array))
    masked_array[masked_array == 0] = np.nan
    masked_array_mean = np.nanmean(masked_array)
    masked_array_std = np.nanstd(masked_array)/np.nanmean(masked_array)

    masked_background = np.multiply(current_slice, otsu_background)
    masked_background[masked_background == 0] = np.nan
    with suppress_stdout():
        masked_background_mean = np.nanmean(masked_background)
    masked_background_std = np.nanstd(masked_background) \
        / np.nanmean(masked_background)

    otsu_fraction = np.count_nonzero(otsu_array)/otsu_array.size
    otsu_snr = masked_array_mean/masked_background_mean

    return otsu_snr, \
        masked_array_std, \
        masked_background_std, \
        otsu_fraction, \
        masked_array_mean, \
        masked_background_mean


def edge_detection(current_slice, current_mask):
    '''
    Function that calculates the chamfer distance between the edge of mask
    to the closest edge-detected pixel in source data
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of source_data stack
    current_mask: array like (_, _)
        Single slice of mask stack
    Outputs
    ----------
    binary_edge_fraction: float
        Fraction of image canvas occupied by pixels categorized as edges
        in source data
    binary_edge_cc_count: int
        Number of connected components contained in edge detected canvas
        for source data
    chamfer_dist
        Chamfer distance between mask edge and closest edge detected in source
        data
    '''
    current_image = sitk.GetImageFromArray(current_slice)
    current_image_mask = sitk.GetImageFromArray(current_mask)

    edge_detection_filter = sitk.SobelEdgeDetectionImageFilter()
    edges_image = edge_detection_filter.Execute(sitk.Cast(current_image,
                                                          sitk.sitkFloat32))
    edges_mask = edge_detection_filter.Execute(sitk.Cast(current_image_mask,
                                                         sitk.sitkFloat32))

    binary_image_filter = sitk.LiThresholdImageFilter()
    binary_image_filter.SetInsideValue(0)
    binary_image_filter.SetOutsideValue(1)
    binary_edges = binary_image_filter.Execute(edges_image)

    try:
        binary_edges_mask = binary_image_filter.Execute(edges_mask)
        binary_edges_array = sitk.GetArrayFromImage(binary_edges)
        binary_edges_mask_array = sitk.GetArrayFromImage(binary_edges_mask)

        connected_component_filter = sitk.ConnectedComponentImageFilter()
        binary_edge_connected_components = connected_component_filter.Execute(
            binary_edges)
        binary_edge_cc_count = connected_component_filter.GetObjectCount() \
            / max(binary_edges_array.shape)

        binary_edge_fraction = np.count_nonzero(binary_edges_array == 1) \
            / binary_edges_array.size
        chamfer_dist = chamfer_distance(
            np.expand_dims(binary_edges_mask_array,
                           axis=0),
            np.expand_dims(binary_edges_array,
                           axis=0)) / binary_edges_mask_array.size
    except RuntimeError:
        binary_edge_fraction = 0
        binary_edge_cc_count = 0
        chamfer_dist = 10

    return binary_edge_fraction, binary_edge_cc_count, chamfer_dist


def intensity_location_check(current_slice):
    '''
    Function that determines the fractional position of the point with the
    largest intensity value in source data
    Parameters
    ----------
    current_slice: array like (_, _)
        Single slice of source data stack
    Outputs
    ----------
    max_loc_horiz: float
        Fractional position of point with largest intensity along the
        horizontal axis. Range [0,1]
    max_loc_vert: float
        Fractional position of point with largest intensity along the
        vertical axis. Range [0,1]
    '''
    max_loc = np.argwhere(current_slice == np.amax(current_slice))
    max_loc_horiz = max_loc[0][0]/current_slice.shape[0]
    max_loc_vert = max_loc[0][1]/current_slice.shape[1]

    return max_loc_horiz, max_loc_vert


def geometry_check(current_mask):
    '''
    Function that grabs information about the geometric qualities of mask
    in a single slice
    Parameters
    ----------
    current_mask: array like (_, _)
        Single slice of mask stack
    Output
    ----------
    roundness: float
        Roundness of mask region in slice
    elongation: float
        Elongation of mask region in slice
    '''
    current_mask_image = sitk.GetImageFromArray(current_mask)

    image_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    image_stats_filter.Execute(sitk.Cast(current_mask_image, sitk.sitkUInt8))

    labels = image_stats_filter.GetLabels()
    no_pixels = []
    for label in labels:
        no_pixels.append(len(image_stats_filter.GetIndexes(label)))
    try:
        primary_label = int(np.argwhere(
            np.array(no_pixels) == np.amax(np.array(no_pixels)))[0][0]+1)
        roundness = image_stats_filter.GetRoundness(primary_label)
        elongation = image_stats_filter.GetElongation(primary_label)
    except RuntimeError:
        roundness = 0
        elongation = 0

    return roundness, elongation


def m_pq(f, p, q):
    '''
    Two-dimensional (p+q)th order moment of image f(x,y)
    where p,q = 0, 1, 2, ...
    '''
    m = 0
    for x in range(0, len(f)):
        for y in range(0, len(f[0])):
            m += ((x+1)**p)*((y+1)**q)*f[x][y]

    return m


def centroid(f):
    '''
    Computes the centroid of image f(x,y)
    '''
    m_00 = m_pq(f, 0, 0)

    return [m_pq(f, 1, 0)/m_00, m_pq(f, 0 ,1)/m_00]


def u_pq(f, p, q):
    '''
    Centroid moment invariant to rotation.
    This function is equivalent to the m_pq but translating the centre of image
    f(x,y) to the centroid.
    '''
    u = 0
    centre = centroid(f)
    for x in range(0, len(f)):
        for y in range(0, len(f[0])):
            u += ((x-centre[0]+1)**p)*((y-centre[1]+1)**q)*f[x][y]

    return u


def hu(f):
    '''
    This function computes Hu's seven invariant moments.
    hu() and associated functions modified from the following:
    https://github.com/adailtonjn68/hu_moments_in_python
    Reference:
    Ming-Kuei Hu, "Visual pattern recognition by moment invariants,"
    in IRE Transactions on Information Theory, vol. 8, no. 2, pp. 179-187,
    February 1962. doi: 10.1109/TIT.1962.1057692
    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1057692&isnumber=22787
    '''
    u_00 = u_pq(f, 0, 0)

    # Scale invariance is obtained by normalization.
    # The normalized central moment is given below
    eta = lambda f, p, q: u_pq(f, p, q)/(u_00**((p+q+2)/2))

    # normalized central moments used to compute Hu's seven moments invariat
    eta_20 = eta(f, 2, 0)
    eta_02 = eta(f, 0, 2)
    eta_11 = eta(f, 1, 1)
    eta_12 = eta(f, 1, 2)
    eta_21 = eta(f, 2, 1)
    eta_30 = eta(f, 3, 0)
    eta_03 = eta(f, 0, 3)

    # Hu's moments are computed below
    phi_1 = eta_20 + eta_02
    phi_2 = 4*eta_11 + (eta_20-eta_02)**2
    phi_3 = (eta_30 - 3*eta_12)**2 + (3*eta_21 - eta_03)**2
    phi_4 = (eta_30 + eta_12)**2 + (eta_21 + eta_03)**2
    phi_5 = (eta_30 - 3*eta_12) \
        * (eta_30 + eta_12) \
        * ((eta_30+eta_12)**2 - 3*(eta_21+eta_03)**2) \
        + (3*eta_21 - eta_03)*(eta_21 + eta_03) \
        * (3*(eta_30 + eta_12) - (eta_21 + eta_03)**2)
    phi_6 = (eta_20 - eta_02) \
        * ((eta_30 + eta_12)**2 - (eta_21 + eta_03)**2) \
        + 4*eta_11*(eta_30 + eta_12)*(eta_21 + eta_03)
    phi_7 = (3*eta_21 - eta_03) \
        * (eta_30 + eta_12) \
        * ((eta_30 + eta_12)**2 - 3*(eta_21 + eta_03)**2) \
        - (eta_30 - 3*eta_12)*(eta_21 + eta_03) \
        * (3*(eta_30 + eta_12)**2 - (eta_21 + eta_03)**2)

    return [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]


def array2samples_distance(array1, array2):
    '''
    Function that calculates the distance between two arrays, one containing
    samples of interest.
    Parameters
    ----------
    array1: array like (_, _)
        The array, size: (num_point, num_feature)
    array2: array like (_, _)
        The samples, size: (num_point, num_feature)
    Output
    ----------
    distances: array like (_, )
        Each entry is the distance from a sample to array1
    '''
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)

    return distances


def chamfer_distance(array1, array2):
    '''
    Calculate the chamfer distance between two arrays
    Parameters
    ----------
    array1: array like (_, _)
        The array, size: (num_point, num_feature)
    array2: array like (_, _)
        The samples, size: (num_point, num_feature)
    Output
    ----------
    dist: float
        Chamfer distance between array1 and array2

    '''
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size

    return dist


def connected_components_count(mask_array):
    '''
    Calculate the number of connected components in a binay array
    Parameters
    ----------
    mask_array: array like (_, _)
        Binary array
    Output
    ----------
    no_connected_components: int
        Number of connected components in binary array
    '''
    mask_slice_img = sitk.Cast(sitk.GetImageFromArray(mask_array),
                               sitk.sitkUInt8)

    connected_components_filter = sitk.ConnectedComponentImageFilter()
    connected_components_labelled = connected_components_filter.Execute(
        mask_slice_img)
    no_connected_components = connected_components_filter.GetObjectCount()

    return no_connected_components


def low_brain_region(source_array, mask_array, source_fn):
    '''
    Supplemental function to quality_check. Detects one common issue not well-
    caught by model quality checks: missing intermediate sized brain region
    in lower-middle part of brain. Accomplishes this by looking at the fraction
    of the image canvas occupied by brain mask by slice in a given image stack.
    If the area decreases in the central slices, those slices are likely going
    to require some manual intervention.
    Parameters
    ----------
    source_array: array like (_, _)
        Single slice of source array stack
    mask_array: array like (_, _)
        Single slice of mask stack
    source_fn: string
        Path to source data file
    Outputs
    ----------
    file_quality_check: DataFrame
        Structure containing information about which slices this supplementary
        method flagged for manual review
    '''
    file_quality_check = pd.DataFrame(columns=['filename',
                                               'slice_index',
                                               'notes'])

    mask_area_array = []
    for i in range(mask_array.shape[0]):
        mask_area_temp, _ = mask_area_check(mask_array[i, :, :],
                                            source_array[i, :, :])
        mask_area_array.append(mask_area_temp)
    peak_area_slice = np.argmax(np.array(mask_area_array))

    prev_check = False
    for i in range(mask_array.shape[0]):
        data_slice = source_array[i, :, :]
        mask_slice = mask_array[i, :, :]
        mask_area, _ = mask_area_check(mask_slice,
                                       data_slice)
        if i != mask_array.shape[0] - 1:
            data_slice_next = source_array[i+1, :, :]
            mask_slice_next = mask_array[i+1, :, :]
            mask_area_next, _ = mask_area_check(mask_slice_next,
                                                data_slice_next)
        if i != 0:
            data_slice_prev = source_array[i-1, :, :]
            mask_slice_prev = mask_array[i-1, :, :]
            mask_area_prev, _ = mask_area_check(mask_slice_prev,
                                                data_slice_prev)
        if i > 0 and i < mask_array.shape[0]-1 and mask_area > 0.25:
            if mask_area < mask_area_next and mask_area < mask_area_prev:
                slice_flag = True
                prev_check = True
                notes = 'Anomalous Mask Area - Low'
            else:
                if prev_check is True:
                    if np.abs(mask_area - mask_area_prev) < 0.01:
                        slice_flag = True
                        notes = 'Anomalous Mask Area - Low'
                    else:
                        prev_check = 0
                        slice_flag = False
                else:
                    slice_flag = False
        else:
            slice_flag = False

        if slice_flag is True:
            slice_df = {'filename': str(source_fn),
                        'slice_index': str(i+1),
                        'notes': str(notes)}
            file_quality_check.append(slice_df, ignore_index=True)

    return file_quality_check


def quality_check(source_array,
                  mask_array,
                  qc_classifier,
                  source_fn,
                  mask_fn,
                  skip_edges):
    '''
    Primary handler for post-inference quality checks. Primary goal is to
    determine which slices are in need of manual review. Two methods.
    1) A classifier trained using expert annotation of mask quality
    2) Bespoke solutions for individual issues not well covered by classifer
    Parameters
    ----------
    source_array: array like (_, _)
        Array corresponding to source data stack
    mask_array: array like (_, _)
        Array corresponding to mask stack
    qc_classifier: sklearn model
        sklearn model trained for slice quality classification
    source_fn: string
        Path to source data
    mask_fn: string
        Path to mask
    skip_edges: bool
        If true, first and last slices will not be included in the output list
        for manual review. If False, they can be included.
    Output
    ----------
    file_quality_check: DataFrame
        Structure containing information about which slices this supplementary
        method flagged for manual review
    '''
    qc_debug = False
    file_quality_check = pd.DataFrame(columns=['filename',
                                               'slice_index',
                                               'notes'])
    stack_quality_check = low_brain_region(source_array, mask_array, source_fn)

    qc_feature_list = []
    for i in range(source_array.shape[0]):
        if skip_edges is True:
            if i == 0 or i == (source_array.shape[0] - 1):
                continue
        current_data_slice = source_array[i, :, :]
        current_mask_slice = mask_array[i, :, :]
        slice_index = (i + 1)/source_array.shape[0]
        notes = 'None'
        try:
            no_connected_components = connected_components_count(
                current_mask_slice)
            binary_edge_fraction, binary_edge_cc_count, chamfer_dist \
                = edge_detection(current_data_slice, current_mask_slice)
            otsu_snr, \
                otsu_std, \
                background_std, \
                otsu_size_frac, \
                foreground_mean_intensity, \
                background_mean_intensity = otsu_snr_check(current_data_slice)
            max_loc_horiz, \
                max_loc_vert = intensity_location_check(current_data_slice)
            roundness, elongation = geometry_check(current_mask_slice)
            current_mask_ratio, \
                current_source_ratio = mask_area_check(current_mask_slice,
                                                       current_data_slice)
            feature_array = np.array([current_mask_ratio,
                                      no_connected_components,
                                      slice_index,
                                      otsu_snr,
                                      otsu_std,
                                      binary_edge_fraction,
                                      binary_edge_cc_count,
                                      chamfer_dist,
                                      max_loc_horiz,
                                      max_loc_vert,
                                      otsu_size_frac,
                                      roundness,
                                      elongation]).reshape(1, -1)
            prediction = (qc_classifier.predict_proba(
                feature_array)[:, 1] >= 0.69).astype(bool)
            if prediction is False:
                notes = 'Model Classified'
        except RuntimeError:
            notes = 'Feature Calculation Failure'
            prediction = False

        if prediction == False:
            slice_df = {'filename': str(source_fn),
                        'slice_index': str(slice_index*source_array.shape[0]),
                        'notes': str(notes)}
            file_quality_check = file_quality_check.append(slice_df,
                                                           ignore_index=True)

        if qc_debug is True:
            qc_feature_list.append(feature_array)

    file_quality_check = pd.merge(file_quality_check, stack_quality_check,
                                  on=['filename', 'slice_index'],
                                  suffixes=['_1', '_2'],
                                  how='outer')

    if qc_debug is True:
        qc_feature_array = np.vstack(qc_feature_list)
        with open('qc_feature_array_uf.npy', 'wb') as f:
            print('Saving Full Calculated Feature List for Quality Checking')
            np.save(f, qc_feature_array)

    return file_quality_check
