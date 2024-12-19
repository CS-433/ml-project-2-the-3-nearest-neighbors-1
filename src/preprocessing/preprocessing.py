import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize

FOREGROUND_THRESHOLD = 0.25

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def value_to_class(v, threshold=0.25):
    df = np.mean(v)
    if df > threshold:
        return 1
    else:
        return 0

def concat_images(img_rgb, img_gray1, img_gray2):

    if img_gray1.shape[1] == 1:
        img_gray1 = np.repeat(img_gray1, 3, axis=1)
    if img_gray2.shape[1] == 1:
        img_gray2 = np.repeat(img_gray2, 3, axis=1)

    concatenated_img = np.concatenate((img_rgb, img_gray1, img_gray2), axis=3)
    concatenated_img = np.transpose(concatenated_img, (0, 2, 3, 1))
    return concatenated_img

def resize_image(image, new_size):
    
    return resize(image, (image.shape[0], new_size, new_size), anti_aliasing=True)

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def mask_to_submission_strings(mask, img_number):
    """Assigns a label to each patch and outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, mask.shape[2], patch_size):
        for i in range(0, mask.shape[1], patch_size):
            patch = mask[:, i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}{}{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, test_masks):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for n, mask in enumerate(test_masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(mask, n+1))