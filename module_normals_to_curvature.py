import numpy as np


def conv_1d(array, kernel_1d):
    """Performs row by row 1D convolutions of the given 2D image with the given 1D kernel."""

    # Input kernel length must be odd
    k_l = len(kernel_1d)
    assert k_l % 2 != 0
    # Convolution is repeat-padded
    extended = np.pad(array, k_l // 2, mode="wrap")
    # Output has same size as input (padded, valid-mode convolution)
    output = np.empty(array.shape)
    for i in range(0, array.shape[0]):
        output[i] = np.convolve(extended[i + (k_l // 2)], kernel_1d, mode="valid")

    return output * -1


def gaussian_kernel(length, sigma):
    """Returns a 1D gaussian kernel of size 'length'."""

    space = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    kernel = np.exp(-0.5 * np.square(space) / np.square(sigma))
    return kernel / np.sum(kernel)


def normalize(np_array):
    """Normalize all elements of the given numpy array to [0,1]"""

    return (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))


def apply(normals_img, blur_radius, progress_callback):
    """Computes a curvature map from the given normal map. 'normals_img' must be a numpy array
    in C,H,W format (with C as RGB). 'blur_radius' must be one of 'SMALLEST', 'SMALLER', 'SMALL',
    'MEDIUM', 'LARGE', 'LARGER', 'LARGEST'."""

    # Convolutions on normal map red & green channels
    if progress_callback is not None:
        progress_callback(0, 4)
    diff_kernel = np.array([-1, 0, 1])
    h_conv = conv_1d(normals_img[0, :, :], diff_kernel)
    if progress_callback is not None:
        progress_callback(1, 4)
    v_conv = conv_1d(-1 * normals_img[1, :, :].T, diff_kernel).T
    if progress_callback is not None:
        progress_callback(2, 4)

    # Sum detected edges
    edges_conv = h_conv + v_conv

    # Blur radius size is proportional to img sizes
    blur_factors = {
        "SMALLEST": 1 / 256,
        "SMALLER": 1 / 128,
        "SMALL": 1 / 64,
        "MEDIUM": 1 / 32,
        "LARGE": 1 / 16,
        "LARGER": 1 / 8,
        "LARGEST": 1 / 4,
    }
    assert blur_radius in blur_factors
    blur_radius_px = int(np.mean(normals_img.shape[1:3]) * blur_factors[blur_radius])

    # If blur radius too small, do not blur
    if blur_radius_px < 2:
        edges_conv = normalize(edges_conv)
        return np.stack([edges_conv, edges_conv, edges_conv])

    # Make sure blur kernel length is odd
    if blur_radius_px % 2 == 0:
        blur_radius_px += 1

    # Blur curvature with separated convolutions
    sigma = blur_radius_px // 8
    if sigma == 0:
        sigma = 1
    g_kernel = gaussian_kernel(blur_radius_px, sigma)
    h_blur = conv_1d(edges_conv, g_kernel)
    if progress_callback is not None:
        progress_callback(3, 4)
    v_blur = conv_1d(h_blur.T, g_kernel).T
    if progress_callback is not None:
        progress_callback(4, 4)

    # Normalize to [0,1]
    curvature = normalize(v_blur)

    # Expand single channel the three channels (RGB)
    return np.stack([curvature, curvature, curvature])
