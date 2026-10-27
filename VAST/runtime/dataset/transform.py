import numpy as np
import random
import torch
from PIL import ImageFilter
from scipy import ndimage
import monai.transforms as mt


def random_rot_flip(img, mask=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    if mask is not None:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
        img = np.flip(img, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        return img, mask
    else:
        img = np.rot90(img, k)
        img = np.flip(img, axis=axis).copy()
        return img


def random_rotate(img, mask=None):
    angle = np.random.randint(-20, 20)
    if mask is not None:
        img = ndimage.rotate(img, angle, order=0, reshape=False)
        mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        return img, mask
    else:
        img = ndimage.rotate(img, angle, order=0, reshape=False)
        return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


train_l_transforms = mt.Compose([
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),  # Vertical flip.
    mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),  # Horizontal flip.

    # Apply the same random affine transform to each image and target mask.
    # Bilinear interpolation is used for images and nearest-neighbor for masks.
    mt.RandAffined(
        keys=["image", "mask"], prob=0.5,
        rotate_range=0.5, translate_range=0.05, scale_range=0.1,
        mode=("bilinear", "nearest"), padding_mode="zeros"
    ),

    # Apply a paired elastic deformation to the image and target mask.
    mt.Rand2DElasticd(
        keys=["image", "mask"], prob=0.4,
        spacing=(32, 32), magnitude_range=(1, 3),
        mode=("bilinear", "nearest"), padding_mode="zeros"
    ),

    # Adjust image contrast without changing the target mask.
    mt.RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2))
])

# Strong augmentations for unlabeled images. Spatial transforms are sampled
# independently; corresponding weak pseudo-labels are not warped by this pipeline.
u_strong_transforms = mt.Compose([
    mt.RandAffined(
        keys=["image"], prob=0.5,
        rotate_range=0.5, translate_range=0.05, scale_range=0.1, mode="bilinear", padding_mode="zeros"
    ),
    mt.Rand2DElasticd(
        keys=["image"], prob=0.4,
        spacing=(32, 32), magnitude_range=(1, 3), mode="bilinear", padding_mode="zeros"
    ),
    mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.3)),

    # Add Gaussian image noise.
    mt.RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.02),

    # Randomly mask image patches.
    mt.RandCoarseDropoutd(keys=["image"], holes=8, spatial_size=32, fill_value=0, prob=0.4)
])


def apply_l_aug(image, mask):
    """
    Apply paired augmentations to a labeled image and target mask.

    Input and output arrays have shape (H, W).
    """
    # Add the channel dimension expected by MONAI.
    data = {
        "image": np.expand_dims(image, 0).astype(np.float32),
        "mask": np.expand_dims(mask, 0).astype(np.float32)
    }

    # Apply the paired transforms.
    data = train_l_transforms(data)

    # Remove the added channel dimension.
    img_out = data["image"][0].numpy()
    mask_out = data["mask"][0].numpy()

    return img_out, mask_out


def apply_u_strong_aug(image):
    """
    Apply strong augmentations to an unlabeled image.

    Input and output arrays have shape (H, W).
    """
    data = {
        "image": np.expand_dims(image, 0).astype(np.float32)
    }

    data = u_strong_transforms(data)

    return data["image"][0].numpy()