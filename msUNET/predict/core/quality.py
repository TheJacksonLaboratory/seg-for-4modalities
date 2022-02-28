import sklearn
import numpy as np
import pandas as pd
import scipy
import sys, os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math
from PIL import ImageDraw, Image
from scipy.spatial import ConvexHull
import joblib
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
	with open(os.devnull, "w") as devnull:
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:  
			yield
		finally:
			sys.stdout = old_stdout

def resample_img_qc(
		imgobj,
		new_spacing=None,
		interpolator=sitk.sitkLinear,
		target_size=[48,48,1],
		revert=False):
	# Function that resamples input images for compatibility with inference model
	# INPTS
	# imgobj: SimpleITK image object corresponding to raw data
	# new_spacing: Muliplicative factor by which image dimensions should be multiplied by. Serves to
	# change image size such that the patch dimension on which the network was trained fits reasonably
	# into the input image. Of the form [horizontal_spacing, vertical_spacing, interlayer_spacing].
	# If images are large, should have elements > 1
	# If images are small should have elements < 1
	# Interpolator: function used to interpolate in the instance new_spacing != [1,1,1]. Default is nearest
	# neighbor as to not introduce new values
	# New_size: declared size for new images. If left as None will calculate the new size automatically
	# baseon on new spacing and old image dimensions/spacing
	# OUTPUTS
	# resampled_imgobj: SimpleITK image object resampled in the desired manner
	resample = sitk.ResampleImageFilter()
	resample.SetInterpolator(interpolator)
	resample.SetOutputDirection(imgobj.GetDirection())
	resample.SetOutputOrigin(imgobj.GetOrigin())
	if not revert:
		if target_size is None:
			orig_img_spacing = np.array(imgobj.GetSpacing())
			resample.SetOutputSpacing(new_spacing)
			orig_size = np.array(imgobj.GetSize(), dtype=np.int)
			orig_spacing = np.array(imgobj.GetSpacing())
			target_size = orig_size * (orig_spacing / new_spacing)
			target_size = np.ceil(target_size).astype(
				np.int)  # Image dimensions are in integers
			target_size = [int(s) for s in target_size]

			new_spacing[2] = orig_img_spacing[2]

			target_size_final = target_size

		if target_size is not None:
			new_spacing = [0, 0, 0]
			orig_img_dims = np.expand_dims(np.array(sitk.GetArrayFromImage(imgobj)),axis=0).shape
			
			orig_img_spacing = np.array(imgobj.GetSpacing())
			target_size[2] = orig_img_dims[0]
			#target_size[1] = int(
			#    np.floor(
			#        (target_size[0] /
			#         orig_img_dims[1]) *
			#        orig_img_dims[2]))
			spacing_ratio_1 = target_size[0] / orig_img_dims[1]
			spacing_ratio_2 = target_size[1] / orig_img_dims[2]
			new_spacing[0] = orig_img_spacing[0] / spacing_ratio_1  # orig 1
			new_spacing[1] = orig_img_spacing[1] / spacing_ratio_2  # orig 2
			resample.SetOutputSpacing(new_spacing)

			# Correct target size image dimensions
			target_size_final = [0, 0, 0]
			target_size_final[0] = target_size[1]
			target_size_final[1] = target_size[0]
			target_size_final[2] = target_size[2]

	if revert:
		resample.SetOutputSpacing(new_spacing)
		target_size_final = target_size

	resample.SetSize(np.array(target_size_final, dtype='int').tolist())

	resampled_array = sitk.GetArrayFromImage(resample.Execute(imgobj))

	return resampled_array

def low_snr_check(current_slice):
	# Function that determines whether a raw data slice should be flagged for manual review due to a low 
		# signal to noise ratio. Does so by comparing mean intensity in center of image to mean intensity
		# in corners. Assumes that the brain is roughly centered in the image
	# INPUTS
	# source_array: numpy array corresponding to entire MRI image scan
	# source_fn: full path to source file
	# low_snr_threshold: Multiplicative factor below which slice will be flagged
	# OUTPUTS
	# snr_check_list: list of slices and the corresponding files that contain them that have been flagged for
	## TODO: Use aim 1 masks to define 'signal' region instead of assuming circle at center
	i,j = np.indices(current_slice.shape)
	# Grab the mean of a 3x3 circle in the center of the image
	center_mean = np.array(current_slice[((i-(current_slice.shape[0]//2))**2 < 9) & ((j-(current_slice.shape[1]//2))**2 < 9)]).mean()
	k,l = 3,3
	# Grab the mean of a 3x3 box on the upper left hand corner of the image
	edge_mean = current_slice[max(0,k-3):k+3,max(0,l-3):l+3].mean()
	# Compare the center mean (brain) to the edge mean (background). If they are very different, warn
	if edge_mean == 0:
		edge_mean = 0.001
	
	return center_mean/edge_mean

def mask_area_check(current_slice,current_slice_source):
	# Function that determines whether a raw data slice should be flagged for manual review due to either a 
		# low or high mask area. If the percentage of pixels in classified as brain in a given slice is outside
		# the interval [0.04,0.8], a flag will be raised.
	total_pixels = current_slice.size
	mask_pixels = (np.asarray(current_slice) > 0).sum()
	source_data_pixels = (np.asarray(current_slice_source) > current_slice_source.mean()).sum()
	mask_ratio = mask_pixels/total_pixels
	source_data_ratio = source_data_pixels/total_pixels  

	return mask_ratio, source_data_ratio

def convex_hull_image(data):
	# Function that calculates and draws the convex hull for a 2D binary image
	region = np.argwhere(data)
	hull = ConvexHull(region)
	verts = [(region[v,0], region[v,1]) for v in hull.vertices]
	img = Image.new('L', data.shape, 0)
	ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
	mask = np.array(img)
	
	return mask.T
	
def solidity_check(current_slice):
	# Function that determines whether a mask should be flagged for manual review due to having a low solidity.
		# For each slice, the percentage of pixels classified as brain is compared to the number of pixels
		# contained in the convex hull containing those pixels. If the count of pixels in the brain is less than
		# 90% of the count in the convex hull, a flag is thrown.
	# Convex hull operation fails if there are fewer than 3 pixels classified as brain
	# In this case, this check is not relevant. Zero-pixel masks are caught by other checks.
	#try:
	current_slice_convex_hull = convex_hull_image(current_slice)
	#except:
		#print('convex hull failure calculation')
		#current_slice_convex_hull = current_slice
	total_pixels = current_slice.size
	mask_pixels = (current_slice == 1).sum()
	convex_hull_pixels = (current_slice_convex_hull == 1).sum()
	slice_solidity = mask_pixels/convex_hull_pixels

def otsu_snr_check(current_slice):
	
	current_image = sitk.GetImageFromArray(current_slice)
	
	otsu_image_filter = sitk.OtsuThresholdImageFilter()
	otsu_image_filter.SetInsideValue(0)
	otsu_image_filter.SetOutsideValue(1)
	
	otsu_mask = otsu_image_filter.Execute(current_image)
	
	otsu_array = sitk.GetArrayFromImage(otsu_mask)
	otsu_background = np.where((otsu_array==0)|(otsu_array==1), otsu_array^1, otsu_array)
	
	masked_array = np.multiply(current_slice,otsu_array)
	masked_array = np.nan_to_num(masked_array)
	masked_array[masked_array == 0] = np.nan
	masked_array_mean = np.nanmean(masked_array)
	masked_array_std = np.nanstd(masked_array)/np.nanmean(masked_array)
	
	masked_background = np.multiply(current_slice,otsu_background)
	masked_background[masked_background == 0] = np.nan
	with suppress_stdout():
		masked_background_mean = np.nanmean(masked_background)
	masked_background_std = np.nanstd(masked_background)/np.nanmean(masked_background)
	
	otsu_fraction = np.count_nonzero(otsu_array)/otsu_array.size
	
	otsu_snr = masked_array_mean/masked_background_mean
	
	return otsu_snr, masked_array_std, masked_background_std, otsu_fraction, masked_array_mean, masked_background_mean 

def edge_detection(current_slice, current_mask):
	
	current_image = sitk.GetImageFromArray(current_slice)
	current_image_mask = sitk.GetImageFromArray(current_mask)
	edge_detection_filter = sitk.SobelEdgeDetectionImageFilter()
	edges_image = edge_detection_filter.Execute(sitk.Cast(current_image,sitk.sitkFloat32))
	edges_mask = edge_detection_filter.Execute(sitk.Cast(current_image_mask,sitk.sitkFloat32))
	
	binary_image_filter = sitk.LiThresholdImageFilter()
	binary_image_filter.SetInsideValue(0)
	binary_image_filter.SetOutsideValue(1)
	binary_edges = binary_image_filter.Execute(edges_image)
	
	try:
		binary_edges_mask = binary_image_filter.Execute(edges_mask)
	
		binary_edges_array = sitk.GetArrayFromImage(binary_edges)
		binary_edges_mask_array = sitk.GetArrayFromImage(binary_edges_mask)

		connected_component_filter = sitk.ConnectedComponentImageFilter()
		binary_edge_connected_components = connected_component_filter.Execute(binary_edges)
		binary_edge_cc_count = connected_component_filter.GetObjectCount()/max(binary_edges_array.shape)

		edge_pixel_count = np.count_nonzero(binary_edges_array == 1)
		binary_edge_fraction = np.count_nonzero(binary_edges_array == 1)/binary_edges_array.size
		chamfer_dist = chamfer_distance(np.expand_dims(binary_edges_mask_array,axis=0), 
										np.expand_dims(binary_edges_array,axis=0))/binary_edges_mask_array.size

		#sitk.WriteImage(binary_edges,'binary_edges.nii')
		#sitk.WriteImage(edges_mask,'binary_li_sobel_mask.nii')
	except RuntimeError:
		binary_edge_fraction = 0
		binary_edge_cc_count = 0
		chamfer_dist = 10
	
	return binary_edge_fraction, binary_edge_cc_count, chamfer_dist

def intensity_location_check(current_slice):
	max_loc = np.argwhere(current_slice == np.amax(current_slice))
	max_loc_horiz = max_loc[0][0]/current_slice.shape[0]
	max_loc_vert = max_loc[0][1]/current_slice.shape[1]

	return max_loc_horiz, max_loc_vert

def geometry_check(current_slice,current_mask):
	current_data_image = sitk.GetImageFromArray(current_slice)
	current_mask_image = sitk.GetImageFromArray(current_mask)
	
	image_stats_filter = sitk.LabelShapeStatisticsImageFilter()
	image_stats_filter.Execute(sitk.Cast(current_mask_image, sitk.sitkUInt8))
	
	label_no = image_stats_filter.GetNumberOfLabels()
	labels = image_stats_filter.GetLabels()
	
	no_pixels = []
	for label in labels:
		no_pixels.append(len(image_stats_filter.GetIndexes(label)))
	try:
		primary_label = int(np.argwhere(np.array(no_pixels) == np.amax(np.array(no_pixels)))[0][0]+1)        
		roundness = image_stats_filter.GetRoundness(primary_label)
		elongation = image_stats_filter.GetElongation(primary_label)
	except:
		roundness = 0
		elongation = 0
	
	return roundness, elongation

def m_pq(f, p, q):
	"""
	Two-dimensional (p+q)th order moment of image f(x,y)
	where p,q = 0, 1, 2, ...
	"""
	m = 0
	# Loop in f(x,y)
	for x in range(0, len(f)):
		for y in range(0, len(f[0])):
			# +1 is used because if it wasn't, the first row and column would
			# be ignored
			m += ((x+1)**p)*((y+1)**q)*f[x][y]
	return m


def centroid(f):
	"""
	Computes the centroid of image f(x,y)
	"""
	m_00 = m_pq(f, 0, 0)
	return [m_pq(f, 1, 0)/m_00, m_pq(f, 0 ,1)/m_00]


def u_pq(f, p, q):
	"""
	Centroid moment invariant to rotation.
	This function is equivalent to the m_pq but translating the centre of image
	f(x,y) to the centroid.
	"""
	u = 0
	centre = centroid(f)
	for x in range(0, len(f)):
		for y in range(0, len(f[0])):
			u += ((x-centre[0]+1)**p)*((y-centre[1]+1)**q)*f[x][y]
	return u


def hu(f):
	"""
	This function computes Hu's seven invariant moments.
	"""
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
	phi_5 = (eta_30 - 3*eta_12)*(eta_30 + eta_12)*((eta_30+eta_12)**2 - 3*(eta_21+eta_03)**2) + (3*eta_21 - eta_03)*(eta_21 + eta_03)*(3*(eta_30 + eta_12) - (eta_21 + eta_03)**2)
	phi_6 = (eta_20 - eta_02)*((eta_30 + eta_12)**2 - (eta_21 + eta_03)**2) + 4*eta_11*(eta_30 + eta_12)*(eta_21 + eta_03)
	phi_7 = (3*eta_21 - eta_03)*(eta_30 + eta_12)*((eta_30 + eta_12)**2 - 3*(eta_21 + eta_03)**2) - (eta_30 - 3*eta_12)*(eta_21 + eta_03)*(3*(eta_30 + eta_12)**2 - (eta_21 + eta_03)**2)

	return [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]

def array2samples_distance(array1, array2):
	"""
	arguments: 
		array1: the array, size: (num_point, num_feature)
		array2: the samples, size: (num_point, num_feature)
	returns:
		distances: each entry is the distance from a sample to array1 
	"""
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
	batch_size, num_point, num_features = array1.shape
	dist = 0
	for i in range(batch_size):
		av_dist1 = array2samples_distance(array1[i], array2[i])
		av_dist2 = array2samples_distance(array2[i], array1[i])
		dist = dist + (av_dist1+av_dist2)/batch_size
	return dist

def connected_components_count(mask_array):
	connected_components_filter = sitk.ConnectedComponentImageFilter()
	mask_slice_img= sitk.Cast(sitk.GetImageFromArray(mask_array),sitk.sitkUInt8)
	connected_components_labelled = connected_components_filter.Execute(mask_slice_img)
	no_connected_components = connected_components_filter.GetObjectCount()

	return no_connected_components

def low_brain_region(source_array, mask_array, source_fn, mask_fn):
	file_quality_check = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])

	mask_area_array = []
	for i in range(mask_array.shape[0]):
		mask_area_temp, _ = mask_area_check(mask_array[i,:,:], source_array[i,:,:])
		mask_area_array.append(mask_area_temp)
	peak_area_slice = np.argmax(np.array(mask_area_array))

	prev_check = False
	for i in range(mask_array.shape[0]):
		data_slice = source_array[i,:,:]
		mask_slice = mask_array[i,:,:]
		mask_area, data_area = mask_area_check(mask_slice, data_slice)
		if i != mask_array.shape[0] - 1:
			data_slice_next = source_array[i+1,:,:]
			mask_slice_next = mask_array[i+1,:,:]
			mask_area_next, data_area_next = mask_area_check(mask_slice_next, data_slice_next)
		if i != 0:
			data_slice_prev = source_array[i-1,:,:]
			mask_slice_prev = mask_array[i-1,:,:]
			mask_area_prev, data_area_prev = mask_area_check(mask_slice_prev, data_slice_prev)
		if i > 0 and i < mask_array.shape[0]-1 and mask_area > 0.25:
			if mask_area < mask_area_next and mask_area < mask_area_prev:
				slice_flag = True
				prev_check = True
				notes='Anomalous Mask Area - Low'
			else:
				if prev_check == True:
					if np.abs(mask_area - mask_area_prev) < 0.01:
						slice_flag = True
						notes='Anomalous Mask Area - Low'
					else:
						prev_check = 0
						slice_flag = False
				else:
					slice_flag = False
		else:
			slice_flag = False

		if slice_flag == True:
			slice_df = {'filename': str(source_fn), 'slice_index': str(i+1), 'notes': str(notes)}
			file_quality_check.append(slice_df, ignore_index=True)

	return file_quality_check



def quality_check(source_array, mask_array,qc_classifier, source_fn, mask_fn, skip_edges):
	qc_debug = True

	file_quality_check = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])
	stack_quality_check = low_brain_region(source_array, mask_array, source_fn, mask_fn)
	qc_feature_list = []

	for i in range(source_array.shape[0]):
		if skip_edges == True:
			if i == 0 or i == (source_array.shape[0] -1):
				continue
		current_data_slice = source_array[i,:,:]
		current_mask_slice = mask_array[i,:,:]
		slice_index = (i + 1)/source_array.shape[0]
		notes = 'None'
		try:
			no_connected_components = connected_components_count(current_mask_slice)
			binary_edge_fraction, binary_edge_cc_count, chamfer_dist = edge_detection(current_data_slice, current_mask_slice)
			otsu_snr, otsu_std, background_std, otsu_size_frac, foreground_mean_intensity, background_mean_intensity = otsu_snr_check(current_data_slice)
			max_loc_horiz, max_loc_vert = intensity_location_check(current_data_slice)
			roundness, elongation = geometry_check(current_data_slice, current_mask_slice)
			current_mask_ratio, current_source_ratio = mask_area_check(current_mask_slice,current_data_slice)
			#basic_snr = low_snr_check(current_data_slice)
			#resampled_data_pixel_list = resample_img_qc(sitk.GetImageFromArray(current_data_slice)).flatten()
			#resampled_mask_pixel_list = resample_img_qc(sitk.GetImageFromArray(current_mask_slice),interpolator=sitk.sitkNearestNeighbor).flatten()
			#solidity = solidity_check(current_mask_slice)
			#hu_array = np.sign(np.array(hu(current_mask_slice)))*np.log(np.absolute(np.array(hu(current_mask_slice))))
			feature_array = np.array([current_mask_ratio, no_connected_components, slice_index,
									otsu_snr, otsu_std, binary_edge_fraction, binary_edge_cc_count,
									chamfer_dist, max_loc_horiz, max_loc_vert, otsu_size_frac,
									roundness, elongation]).reshape(1,-1)
			prediction = (qc_classifier.predict_proba(feature_array)[:,1] >= 0.69).astype(bool)
			print(feature_array)
			print(prediction)
			if prediction == False:
				notes = 'Model Classified'		
		except:
			notes = 'Feature Calculation Failure'
			prediction = False

		if prediction == False:
			slice_df = {'filename': str(source_fn), 'slice_index': str(slice_index*source_array.shape[0]), 'notes': str(notes)}
			file_quality_check = file_quality_check.append(slice_df, ignore_index=True)

		if qc_debug == True:
			qc_feature_list.append(feature_array)


	file_quality_check = pd.merge(file_quality_check, stack_quality_check, 
								  on=['filename','slice_index'],
								  suffixes=['_1','_2'],
								  how='outer')

	if qc_debug == True:
		qc_feature_array = np.vstack(qc_feature_list)
		with open('qc_feature_array_uf.npy', 'wb') as f:
			print('Saving Full Calculated Feature List for Quality Checking')
			np.save(f, qc_feature_array)

	return file_quality_check











