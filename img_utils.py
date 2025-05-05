from __future__ import print_function, division, absolute_import

import numpy as np
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage import gaussian_filter
from keras import backend as K

import os
import time
from PIL import Image
import imageio

# Configuration
_image_scale_multiplier = 1
img_size = 128 * _image_scale_multiplier
stride = 64 * _image_scale_multiplier

assert (img_size ** 2) % (stride ** 2) == 0, (
    "Number of images generated from strided subsample of the image needs to be a positive integer. "
    "Change stride such that (img_size ** 2) / (stride ** 2) is a positive integer."
)

# Paths
input_path = r"D:\Yue\Documents\Datasets\train2014\train2014\\"
validation_path = r"val_images/"
validation_set5_path = validation_path + "set5/"
validation_set14_path = validation_path + "set14/"
base_dataset_dir = os.path.expanduser("~") + "/Image Super Resolution Dataset/"
output_path = base_dataset_dir + "train_images/train/"
validation_output_path = base_dataset_dir + r"train_images/validation/"

# Ensure directories exist
os.makedirs(output_path + "X/", exist_ok=True)
os.makedirs(output_path + "y/", exist_ok=True)

# === Utility Functions ===

def resize_image(image, size):
    return np.array(Image.fromarray(image).resize(size, resample=Image.BICUBIC))

def subimage_generator(img, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, img_size, stride):
            for y in range(0, img_size, stride):
                subimage = img[x : x + patch_size, y : y + patch_size, :]
                yield subimage

def image_count():
    return len(os.listdir(output_path + "X/"))

def val_image_count():
    return len(os.listdir(validation_output_path + "X/"))

def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    height, width = x.shape[:2]
    if upscale:
        x = resize_image(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches

def combine_patches(in_patches, out_shape, scale):
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon

# === Main Image Transformation Function ===

def transform_images_temp(directory, output_directory, scaling_factor=2, max_nb_images=-1, true_upscale=False,
                          id_advance=0):
    index = 1
    os.makedirs(output_directory + "X/", exist_ok=True)
    os.makedirs(output_directory + "y/", exist_ok=True)

    files = [file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    nb_images = len(files)

    if max_nb_images != -1:
        print(f"Transforming {max_nb_images} images.")
    else:
        print(f"Transforming {nb_images} images.")
        assert max_nb_images <= nb_images or max_nb_images == -1

    if nb_images == 0:
        print("No training images found. Please check your path.")
        exit()

    for file in files:
        img = imageio.imread(os.path.join(directory, file))
        img = resize_image(img, (img_size, img_size))

        hr_patch_size = 64
        lr_patch_size = 32
        nb_hr_images = (img_size ** 2) // (stride ** 2)

        hr_samples = np.empty((nb_hr_images, hr_patch_size, hr_patch_size, 3))
        image_subsample_iterator = subimage_generator(img, stride, hr_patch_size, nb_hr_images)
        stride_range = int(np.sqrt(nb_hr_images))

        i = 0
        for j in range(stride_range):
            for k in range(stride_range):
                hr_samples[i, :, :, :] = next(image_subsample_iterator)
                i += 1

        t1 = time.time()
        for i in range(nb_hr_images):
            ip = hr_samples[i]
            imageio.imwrite(f"{output_directory}/y/{index + id_advance}_{i + 1}.png", ip)

            op = resize_image(ip, (lr_patch_size, lr_patch_size))
            if not true_upscale:
                op = resize_image(op, (hr_patch_size, hr_patch_size))

            imageio.imwrite(f"{output_directory}/X/{index + id_advance}_{i + 1}.png", op)

        print("Finished image %d in time %.2f seconds. (%s)" % (index + id_advance, time.time() - t1, file))
        index += 1

        if max_nb_images > 0 and index >= max_nb_images:
            print("Transformed maximum number of images.")
            break

    print("Images transformed. Saved at directory: %s" % output_directory)

# === Placeholder: image_generator() (incomplete from original)
# Add if needed.

