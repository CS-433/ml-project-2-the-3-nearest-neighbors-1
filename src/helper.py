import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize

FOREGROUND_THRESHOLD = 0.25  # Threshold for classifying a patch as foreground or background.

def load_image(infilename):
    """Loads an image from a file."""
    data = mpimg.imread(infilename)
    return data

def img_crop(im, w, h):
    """
    Crops an image into patches of size w x h.
    
    Parameters:
    - im: Input image as a NumPy array.
    - w: Width of the patches.
    - h: Height of the patches.
    
    Returns:
    - list_patches: List of cropped patches.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3  # Check if the image is 2D (grayscale).
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class(v, threshold=0.25):
    """
    Maps a patch value to a binary class based on a threshold.
    
    Parameters:
    - v: Patch values (e.g., mean intensity).
    - threshold: Value above which the patch is classified as foreground.
    
    Returns:
    - 1 if the patch mean exceeds the threshold, else 0.
    """
    df = np.mean(v)
    return 1 if df > threshold else 0

def concat_images(img_rgb, img_gray1, img_gray2):
    """
    Concatenates an RGB image with two grayscale images along the channel axis.
    
    Parameters:
    - img_rgb: RGB image.
    - img_gray1: First grayscale image.
    - img_gray2: Second grayscale image.
    
    Returns:
    - Concatenated image with all channels combined.
    """
    if img_gray1.shape[1] == 1:
        img_gray1 = np.repeat(img_gray1, 3, axis=1)
    if img_gray2.shape[1] == 1:
        img_gray2 = np.repeat(img_gray2, 3, axis=1)

    concatenated_img = np.concatenate((img_rgb, img_gray1, img_gray2), axis=3)
    concatenated_img = np.transpose(concatenated_img, (0, 2, 3, 1))  # Rearrange axes to image format.
    return concatenated_img

def resize_image(image, new_size):
    """
    Resizes an image to a new size using anti-aliasing.
    
    Parameters:
    - image: Input image.
    - new_size: Desired size for the new image (new_size x new_size).
    
    Returns:
    - Resized image.
    """
    return resize(image, (image.shape[0], new_size, new_size), anti_aliasing=True)

def patch_to_label(patch):
    """
    Assigns a binary label to a patch based on the foreground threshold.
    
    Parameters:
    - patch: Input patch.
    
    Returns:
    - 1 if the mean patch value exceeds the threshold, else 0.
    """
    df = np.mean(patch)
    return 1 if df > FOREGROUND_THRESHOLD else 0

def mask_to_submission_strings(mask, img_number):
    """
    Converts a mask to submission strings for each patch.
    
    Parameters:
    - mask: Input mask (binary image).
    - img_number: Index of the image.
    
    Yields:
    - Strings in the format required for submission.
    """
    patch_size = 16  # Size of each patch.
    for j in range(0, mask.shape[2], patch_size):
        for i in range(0, mask.shape[1], patch_size):
            patch = mask[:, i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, test_masks):
    """
    Converts a list of masks into a CSV submission file.
    
    Parameters:
    - submission_filename: Path to the output submission file.
    - test_masks: List of masks to be converted.
    
    Outputs:
    - Writes the submission strings to the specified file.
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')  # Write CSV header.
        for n, mask in enumerate(test_masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(mask, n+1))