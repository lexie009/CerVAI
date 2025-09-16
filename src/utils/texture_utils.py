import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops
from skimage.filters import sobel
from scipy.stats import entropy


def compute_glcm_features(image, distances=[1, 5, 10, 15], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    if image.ndim == 3:
        image_gray = rgb2gray(image)
    else:
        image_gray = image

    image_gray = (image_gray * 255).astype(np.uint8) if image_gray.max() <= 1 else image_gray.astype(np.uint8)

    glcm = graycomatrix(image_gray,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean()
    }
    return features


def compute_lbp_entropy(image, radius=1, n_points=8):
    if image.ndim == 3:
        image_gray = rgb2gray(image)
    else:
        image_gray = image

    image_gray = (image_gray * 255).astype(np.uint8) if image_gray.max() <= 1 else image_gray.astype(np.uint8)
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return entropy(hist + 1e-10)  # add epsilon to avoid log(0)


def compute_boundary_roughness(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_bin = (mask > 0).astype(np.uint8)

    edges = sobel(mask_bin)
    labeled = label(mask_bin)
    props = regionprops(labeled)

    if not props:
        return 0.0

    region = max(props, key=lambda x: x.area)
    perimeter = region.perimeter
    area = region.area
    return (perimeter ** 2) / (area + 1e-6)  # roughness index


def compute_color_entropy(image, channel='hue', bins=32):
    hsv = rgb2hsv(image)
    if channel == 'hue':
        ch_data = hsv[:, :, 0]
    elif channel == 'saturation':
        ch_data = hsv[:, :, 1]
    elif channel == 'value':
        ch_data = hsv[:, :, 2]
    else:
        raise ValueError("Unsupported channel. Choose from 'hue', 'saturation', 'value'")

    hist, _ = np.histogram(ch_data.ravel(), bins=bins, range=(0, 1), density=True)
    return entropy(hist + 1e-10)


def compute_texture_score(image, mask=None,
                           lambda_glcm=0.25, lambda_lbp=0.25,
                           lambda_rough=0.25, lambda_color=0.25):
    glcm_feats = compute_glcm_features(image)
    lbp_entropy = compute_lbp_entropy(image)
    roughness = compute_boundary_roughness(mask) if mask is not None else 0.0
    color_entropy = compute_color_entropy(image, channel='hue')

    score = (lambda_glcm * glcm_feats['contrast'] +
             lambda_lbp * lbp_entropy +
             lambda_rough * roughness +
             lambda_color * color_entropy)

    return {
        'score': score,
        'glcm_contrast': glcm_feats['contrast'],
        'lbp_entropy': lbp_entropy,
        'roughness': roughness,
        'color_entropy': color_entropy
    }
