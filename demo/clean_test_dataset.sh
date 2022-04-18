#!/bin/bash
find ./test_dataset/ -maxdepth 3 -type f -name "*_mask*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*_likelihood*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*intensity_by_slice*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*z_axis*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*y_axis*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*backup*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*n4b*.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "quality_check.csv" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "input_log.txt" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "segmentation_log.txt" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*segmentation.nii*" -delete
find ./test_dataset/ -maxdepth 3 -type f -name "*segmentation_mask.nii*" -delete