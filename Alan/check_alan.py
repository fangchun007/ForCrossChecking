import cv2
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np


def flip_image(image, flag0):
    if flag0:
        return image[:, ::-1, :]
    return image
def increase_contrast(image, flag1):
    if flag1:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # split the LAB image to different channels
        l, a, b = cv2.split(lab)
        # apply clahe to L-channel
        clipLimit = np.random.uniform(1.2,1.8)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        cl = clahe.apply(l)
        # merge the clahe enhanced l-channel with a, b channels
        image = cv2.merge((cl,a,b))
        # LAB2RGB
        #result = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)/255
        #result = result.astype('float32')
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image
def random_brightness(image, flag2):
    if flag2:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        scale = np.random.uniform(low=0.5, high=1)
        # to mimic bad weather
        # by observation, scaling in the range [0.5,0.9] won't cause image distortion
        image_hsv[:,:,2] = image_hsv[:,:,2] * scale
        image = cv2.cvtColor(image_hsv,cv2.COLOR_HSV2RGB)
        return image
    return image
def blur_image(image, flag3):
    if flag3:
        #kernel_size = np.random.choice([3,5,7])
        return cv2.blur(image, (3,3))
    return image
def augment_pipeline(image, flags):
    # random flip
    image = flip_image(image, flags[0])
    # random contrast
    image = increase_contrast(image, flags[1])
    # random brightness
    image = random_brightness(image, flags[2])
    #image = blur_image(image, flags[3])
    return image

image = scipy.misc.imread('um_000000.png')
print("Original image ('um_000000.png') info:\n    img_dtype: {}\n    max_val: {}\n    min_val: {}".format(image.dtype, np.max(image), np.min(image)))
shape = image.shape

print("\nCheck: flip image")
image_flip = flip_image(image, 1)
print("    flipped image's info:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_flip.dtype, np.max(image_flip), np.min(image_flip)))
image_flip_n = (image_flip - np.ones(shape)*128)/256
print("    After normalized by (image_flip - np.ones(shape)*128)/256:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_flip_n.dtype, np.max(image_flip_n), np.min(image_flip_n)))

print("\nCheck: increase image contrast")
image_contrast = increase_contrast(image, 1)
print("    contrast increased image's info:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_flip.dtype, np.max(image_contrast), np.min(image_contrast)))
image_contrast_n = (image_contrast - np.ones(shape)*128)/256
print("    After normalized by (image_contrast - np.ones(shape)*128)/256:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_contrast_n.dtype, np.max(image_contrast_n), np.min(image_contrast_n)))

print("\nCheck: increase image brightness")
image_brightness = random_brightness(image, 1)
print("    brightness increased image's info:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_brightness.dtype, np.max(image_brightness), np.min(image_brightness)))
image_brightness_n = (image_brightness - np.ones(shape)*128)/256
print("    After normalized by (image_brightness - np.ones(shape)*128)/256:")
print("        img_dtype: {}\n        max_val: {}\n        min_val: {}".format(image_brightness_n.dtype, np.max(image_brightness_n), np.min(image_brightness_n)))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(image_flip)
plt.title('Flipped image')

plt.subplot(2,2,3)
plt.imshow(image_contrast)
plt.title('Contrast increased')

plt.subplot(2,2,4)
plt.imshow(image_brightness)
plt.title('Brightness increased')

plt.show()

















