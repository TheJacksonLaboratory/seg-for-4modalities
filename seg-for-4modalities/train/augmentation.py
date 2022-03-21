import SimpleITK as sitk
import numpy as np
import random


def full_augmentation(
        data_path,
        mask_path,
        data_filenames,
        mask_filenames,
        enable_rotation,
        enable_affine,
        enable_noise,
        enable_contrast_change,
        augmentation_threshold):
    # Function handling data augmentation for mouse MRI datasets. Allows for four different types of augmentation:
    # rotation, affine transforms, additive gaussian noise, and contrast adjustments. Governed by a single parameter,
    # augmentation threshold, which determines both the frequency with which images will be augmented and the number of
    # augmentations to be applied to each. In each case, the intensity is
    # random along a small interval
    if any("augmented" in filenames for filenames in mask_filenames):
        return
    for i in range(0, len(data_filenames)):
        if random.random() < (1 - augmentation_threshold):
            continue
        # Define Reference Images
        data_image = sitk.ReadImage(data_path + '/' + data_filenames[i])
        mask_image = sitk.ReadImage(mask_path + '/' + mask_filenames[i])
        # Do a noddi check, if it's noddi, cull the extra dimension
        image_spacing = data_image.GetSpacing()
        noddi_check_array = sitk.GetArrayFromImage(data_image)
        if len(noddi_check_array.shape) > 3:
            training_array = sitk.GetArrayFromImage(data_image)
            noddi_check_array = noddi_check_array[7, :, :, :]
            training_img_noddi = sitk.GetImageFromArray(noddi_check_array)
            training_img_noddi.SetSpacing(image_spacing)
            data_image = training_img_noddi
        # Make origin consistent such that transforms do not alter
        # correspondence between mask and image
        data_image.SetOrigin((0, 0, 0))
        mask_image.SetOrigin((0, 0, 0))
        data_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        mask_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        augmentation_count = 0

        if enable_rotation:
            if random.random() > augmentation_threshold:
                # Define the transform
                rotation_transform = sitk.Euler3DTransform()
                rotation_transform.SetTranslation(
                    (random.random() - 0.5, random.random() - 0.5, random.random() - 0.5))
                rotation_transform.SetRotation(angleX=(random.random() - 0.5) * (np.pi) / 50, angleY=(
                    random.random() - 0.5) * (np.pi) / 50, angleZ=(random.random() - 0.5) * (np.pi) / 50)
                # Define the interpolators
                interpolator_data = sitk.sitkLinear
                interpolator_mask = sitk.sitkLinear
                # Define the resampling for the data
                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator(interpolator_data)
                resample.SetOutputDirection(data_image.GetDirection())
                resample.SetOutputOrigin(data_image.GetOrigin())
                resample.SetOutputSpacing(data_image.GetSpacing())
                resample.SetSize(data_image.GetSize())
                resample.SetTransform(rotation_transform)
                # Define the resampling for the mask
                resample_mask = sitk.ResampleImageFilter()
                resample_mask.SetInterpolator(interpolator_mask)
                resample_mask.SetOutputDirection(mask_image.GetDirection())
                resample_mask.SetOutputOrigin(mask_image.GetOrigin())
                resample_mask.SetOutputSpacing(mask_image.GetSpacing())
                resample_mask.SetSize(mask_image.GetSize())
                resample_mask.SetTransform(rotation_transform)
                # Execute the transform
                resampled_data_image = resample.Execute(data_image)
                resampled_mask_image = resample_mask.Execute(mask_image)
                # Make an array out of the mask for binary classification
                resampled_mask_array = sitk.GetArrayFromImage(
                    resampled_mask_image)
                super_threshold_indicies = resampled_mask_array > (0.6)
                sub_threshold_indicies = resampled_mask_array <= (0.6)
                resampled_mask_array[super_threshold_indicies] = 1
                resampled_mask_array[sub_threshold_indicies] = 0
                binary_resampled_mask_image = sitk.GetImageFromArray(
                    resampled_mask_array)
                # Copy metadata back over to new images
                binary_resampled_mask_image.CopyInformation(mask_image)
                resampled_data_image.CopyInformation(data_image)

                data_image = resampled_data_image
                mask_image = binary_resampled_mask_image

                augmentation_count += 1

        if enable_affine:
            if random.random() > augmentation_threshold:
                # Define the transform
                affine_transform = sitk.AffineTransform(3)
                affine_transform.Scale((1 + (random.random() - 0.5) * .1,
                                        1 + (random.random() - 0.5) * .1,
                                        1 + (random.random() - 0.5) * .1))
                shear_axis_1 = random.randrange(1, 3)
                shear_axis_2 = random.randrange(1, 3)
                while shear_axis_1 == shear_axis_2:
                    shear_axis_2 = random.randrange(1, 3)
                affine_transform.Shear(
                    shear_axis_1, shear_axis_2, (random.random()) * .1)
                # Define the interpolators
                interpolator_data = sitk.sitkLinear
                interpolator_mask = sitk.sitkLinear
                # Define the resampling for the data
                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator(interpolator_data)
                resample.SetOutputDirection(data_image.GetDirection())
                resample.SetOutputOrigin(data_image.GetOrigin())
                resample.SetOutputSpacing(data_image.GetSpacing())
                resample.SetSize(data_image.GetSize())
                resample.SetTransform(affine_transform)
                # Define the resampling for the mask
                resample_mask = sitk.ResampleImageFilter()
                resample_mask.SetInterpolator(interpolator_mask)
                resample_mask.SetOutputDirection(mask_image.GetDirection())
                resample_mask.SetOutputOrigin(mask_image.GetOrigin())
                resample_mask.SetOutputSpacing(mask_image.GetSpacing())
                resample_mask.SetSize(mask_image.GetSize())
                resample_mask.SetTransform(affine_transform)
                # Execute the transform
                resampled_data_image = resample.Execute(data_image)
                resampled_mask_image = resample_mask.Execute(mask_image)
                # Make an array out of the mask for binary classification
                resampled_mask_array = sitk.GetArrayFromImage(
                    resampled_mask_image)
                super_threshold_indicies = resampled_mask_array > (0.6)
                sub_threshold_indicies = resampled_mask_array <= (0.6)
                resampled_mask_array[super_threshold_indicies] = 1
                resampled_mask_array[sub_threshold_indicies] = 0
                binary_resampled_mask_image = sitk.GetImageFromArray(
                    resampled_mask_array)
                # Copy metadata back over to new images
                binary_resampled_mask_image.CopyInformation(mask_image)
                resampled_data_image.CopyInformation(data_image)

                data_image = resampled_data_image
                mask_image = binary_resampled_mask_image

                augmentation_count += 1

        if enable_noise:
            if random.random() > augmentation_threshold:
                pixelID_data = data_image.GetPixelID()
                pixelID_mask = mask_image.GetPixelID()
                # Define transform - additive gaussian noise - no change to
                # mask required
                gaussian = sitk.AdditiveGaussianNoiseImageFilter()
                gaussian.SetStandardDeviation(3)
                noisy_img = gaussian.Execute(data_image)
                # define caster to ensure output image is of the correct type
                caster = sitk.CastImageFilter()
                caster.SetOutputPixelType(pixelID_data)
                final_noisy_image = caster.Execute(noisy_img)

                data_image = final_noisy_image

                augmentation_count += 1

        if enable_contrast_change:
            if random.random() > augmentation_threshold:
                pixelID_data = data_image.GetPixelID()
                pixelID_mask = mask_image.GetPixelID()
                # Define transform - histogram equalization with the intention
                # of reducing contrast - no change to mask required
                histfilter = sitk.AdaptiveHistogramEqualizationImageFilter()
                histfilter.SetAlpha(0.9)
                histfilter.SetBeta(0.5)
                histfilter_img = histfilter.Execute(data_image)
                caster = sitk.CastImageFilter()
                caster.SetOutputPixelType(pixelID_data)
                final_histfilter_image = caster.Execute(histfilter_img)

                data_image = final_histfilter_image

            augmentation_count += 1

        if augmentation_count > 0:
            index = data_filenames[i].find('.n')
            mask_index = mask_filenames[i].find('.n')
            temp_data_filename = '/' + data_filenames[i][:index] + 'augmented' + str(
                augmentation_count) + data_filenames[i][index:]
            temp_mask_filename = '/' + mask_filenames[i][:mask_index] + 'augmented' + str(
                augmentation_count) + mask_filenames[i][mask_index:]
            sitk.WriteImage(data_image, data_path + temp_data_filename)
            sitk.WriteImage(mask_image, mask_path + temp_mask_filename)


def rotation_augmentation(
        data_path,
        mask_path,
        data_filenames,
        mask_filenames):
    # Deprecated
    # Function that handles rotation augmentation for mouse MRI datasets
    if any("rotated" in filenames for filenames in mask_filenames) or any(
            "rotated" in filenames for filenames in data_filenames):
        return
    # Apply transformation/s to each mask and training image
    for i in range(0, len(data_filenames)):
        # Define Reference Images
        data_image = sitk.ReadImage(data_path + '/' + data_filenames[i])
        mask_image = sitk.ReadImage(mask_path + '/' + mask_filenames[i])
        # Do a noddi check, if it's noddi, cull the extra dimension
        image_spacing = data_image.GetSpacing()
        noddi_check_array = sitk.GetArrayFromImage(data_image)
        if len(noddi_check_array.shape) > 3:
            training_array = sitk.GetArrayFromImage(data_image)
            noddi_check_array = noddi_check_array[7, :, :, :]
            training_img_noddi = sitk.GetImageFromArray(noddi_check_array)
            training_img_noddi.SetSpacing(image_spacing)
            data_image = training_img_noddi
        # Make origin consistent such that transforms do not alter
        # correspondence between mask and image
        data_image.SetOrigin((0, 0, 0))
        mask_image.SetOrigin((0, 0, 0))
        data_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        mask_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        # Define the transform
        rotation_transform = sitk.Euler3DTransform()
        rotation_transform.SetTranslation(
            (random.random() - 0.5, random.random() - 0.5, random.random() - 0.5))
        rotation_transform.SetRotation(angleX=(random.random() - 0.5) * (np.pi) / 50, angleY=(
            random.random() - 0.5) * (np.pi) / 50, angleZ=(random.random() - 0.5) * (np.pi) / 50)
        # Define the interpolators
        interpolator_data = sitk.sitkLinear
        interpolator_mask = sitk.sitkLinear
        # Define the resampling for the data
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator_data)
        resample.SetOutputDirection(data_image.GetDirection())
        resample.SetOutputOrigin(data_image.GetOrigin())
        resample.SetOutputSpacing(data_image.GetSpacing())
        resample.SetSize(data_image.GetSize())
        resample.SetTransform(rotation_transform)
        # Define the resampling for the mask
        resample_mask = sitk.ResampleImageFilter()
        resample_mask.SetInterpolator(interpolator_mask)
        resample_mask.SetOutputDirection(mask_image.GetDirection())
        resample_mask.SetOutputOrigin(mask_image.GetOrigin())
        resample_mask.SetOutputSpacing(mask_image.GetSpacing())
        resample_mask.SetSize(mask_image.GetSize())
        resample_mask.SetTransform(rotation_transform)
        # Execute the transform
        resampled_data_image = resample.Execute(data_image)
        resampled_mask_image = resample_mask.Execute(mask_image)
        # Make an array out of the mask for binary classification
        resampled_mask_array = sitk.GetArrayFromImage(resampled_mask_image)
        super_threshold_indicies = resampled_mask_array > (0.6)
        sub_threshold_indicies = resampled_mask_array <= (0.6)
        resampled_mask_array[super_threshold_indicies] = 1
        resampled_mask_array[sub_threshold_indicies] = 0
        binary_resampled_mask_image = sitk.GetImageFromArray(
            resampled_mask_array)
        # Copy metadata back over to new images
        binary_resampled_mask_image.CopyInformation(mask_image)
        resampled_data_image.CopyInformation(data_image)
        # Write images to their respective locations
        index = data_filenames[i].find('.n')
        mask_index = mask_filenames[i].find('.n')
        temp_data_filename = '/' + \
            data_filenames[i][:index] + 'rotated' + data_filenames[i][index:]
        temp_mask_filename = '/' + \
            mask_filenames[i][:mask_index] + 'rotated' + mask_filenames[i][mask_index:]
        sitk.WriteImage(resampled_data_image, data_path + temp_data_filename)
        sitk.WriteImage(
            binary_resampled_mask_image,
            mask_path + temp_mask_filename)


def affine_augmentation(data_path, mask_path, data_filenames, mask_filenames):
    # Deprecated
    # Function that handles affine augmentations for mouse MRI data. Includes
    # translations and shear
    if any("affine" in filenames for filenames in mask_filenames) or any(
            "affine" in filenames for filenames in data_filenames):
        return
    # Only interested in applying affine transformations to dti and noddi
    # images. Cut anatomical and fmri from the set
    dti_mask_filenames = [match for match in mask_filenames if "dti" in match]
    noddi_mask_filenames = [
        match for match in mask_filenames if "noddi" in match]
    regwarp_mask_filenames = [
        match for match in mask_filenames if "regwarp" in match]
    fmri_mask_filenames = [
        match for match in mask_filenames if "fmri" in match]
    dti_data_filenames = [match for match in data_filenames if "dti" in match]
    noddi_data_filenames = [
        match for match in data_filenames if "noddi" in match]
    regwarp_data_filenames = [
        match for match in data_filenames if "regwarp" in match]
    fmri_data_filenames = [
        match for match in data_filenames if "fmri" in match]
    mask_filenames = dti_mask_filenames + noddi_mask_filenames + \
        regwarp_mask_filenames + fmri_mask_filenames
    data_filenames = dti_data_filenames + noddi_data_filenames + \
        regwarp_data_filenames + fmri_data_filenames
    for i in range(0, len(data_filenames)):
        # Define Reference Images
        data_image = sitk.ReadImage(data_path + '/' + data_filenames[i])
        mask_image = sitk.ReadImage(mask_path + '/' + mask_filenames[i])
        # Do a noddi check, if it's noddi, cull the extra dimension
        image_spacing = data_image.GetSpacing()
        noddi_check_array = sitk.GetArrayFromImage(data_image)
        if len(noddi_check_array.shape) > 3:
            training_array = sitk.GetArrayFromImage(data_image)
            noddi_check_array = noddi_check_array[7, :, :, :]
            training_img_noddi = sitk.GetImageFromArray(noddi_check_array)
            training_img_noddi.SetSpacing(image_spacing)
            data_image = training_img_noddi
        # Make origin consistent such that transforms do not alter
        # correspondence between mask and image
        data_image.SetOrigin((0, 0, 0))
        mask_image.SetOrigin((0, 0, 0))
        data_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        mask_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        # Define the transform
        affine_transform = sitk.AffineTransform(3)
        affine_transform.Scale((1 + (random.random() - 0.5) * .1,
                                1 + (random.random() - 0.5) * .1,
                                1 + (random.random() - 0.5) * .1))
        shear_axis_1 = random.randrange(1, 3)
        shear_axis_2 = random.randrange(1, 3)
        while shear_axis_1 == shear_axis_2:
            shear_axis_2 = random.randrange(1, 3)
        affine_transform.Shear(
            shear_axis_1,
            shear_axis_2,
            (random.random()) * .1)
        # Define the interpolators
        interpolator_data = sitk.sitkLinear
        interpolator_mask = sitk.sitkLinear
        # Define the resampling for the data
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator_data)
        resample.SetOutputDirection(data_image.GetDirection())
        resample.SetOutputOrigin(data_image.GetOrigin())
        resample.SetOutputSpacing(data_image.GetSpacing())
        resample.SetSize(data_image.GetSize())
        resample.SetTransform(affine_transform)
        # Define the resampling for the mask
        resample_mask = sitk.ResampleImageFilter()
        resample_mask.SetInterpolator(interpolator_mask)
        resample_mask.SetOutputDirection(mask_image.GetDirection())
        resample_mask.SetOutputOrigin(mask_image.GetOrigin())
        resample_mask.SetOutputSpacing(mask_image.GetSpacing())
        resample_mask.SetSize(mask_image.GetSize())
        resample_mask.SetTransform(affine_transform)
        # Execute the transform
        resampled_data_image = resample.Execute(data_image)
        resampled_mask_image = resample_mask.Execute(mask_image)
        # Make an array out of the mask for binary classification
        resampled_mask_array = sitk.GetArrayFromImage(resampled_mask_image)
        super_threshold_indicies = resampled_mask_array > (0.6)
        sub_threshold_indicies = resampled_mask_array <= (0.6)
        resampled_mask_array[super_threshold_indicies] = 1
        resampled_mask_array[sub_threshold_indicies] = 0
        binary_resampled_mask_image = sitk.GetImageFromArray(
            resampled_mask_array)
        # Copy metadata back over to new images
        binary_resampled_mask_image.CopyInformation(mask_image)
        resampled_data_image.CopyInformation(data_image)

        # Write images to their respective locations
        index = data_filenames[i].find('.n')
        mask_index = mask_filenames[i].find('.n')
        temp_data_filename = '/' + \
            data_filenames[i][:index] + 'affine' + data_filenames[i][index:]
        temp_mask_filename = '/' + \
            mask_filenames[i][:mask_index] + 'affine' + mask_filenames[i][mask_index:]
        sitk.WriteImage(resampled_data_image, data_path + temp_data_filename)
        sitk.WriteImage(
            binary_resampled_mask_image,
            mask_path + temp_mask_filename)
