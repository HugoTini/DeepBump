import numpy as np


def normals_to_grad(normals_img):
    return (normals_img[0] - 0.5) * 2, (normals_img[1] - 0.5) * 2


def copy_flip(grad_x, grad_y):
    """Concat 4 flipped copies of input gradients (makes them wrap).
    Output is twice bigger in both dimensions."""

    grad_x_top = np.hstack([grad_x, -np.flip(grad_x, axis=1)])
    grad_x_bottom = np.hstack([np.flip(grad_x, axis=0), -np.flip(grad_x)])
    new_grad_x = np.vstack([grad_x_top, grad_x_bottom])

    grad_y_top = np.hstack([grad_y, np.flip(grad_y, axis=1)])
    grad_y_bottom = np.hstack([-np.flip(grad_y, axis=0), -np.flip(grad_y)])
    new_grad_y = np.vstack([grad_y_top, grad_y_bottom])

    return new_grad_x, new_grad_y


def frankot_chellappa(grad_x, grad_y, progress_callback=None):
    """Frankot-Chellappa depth-from-gradient algorithm."""

    if progress_callback is not None:
        progress_callback(0,3)

    rows, cols = grad_x.shape

    rows_scale = (np.arange(rows) - (rows // 2 + 1)) / (rows - rows % 2)
    cols_scale = (np.arange(cols) - (cols // 2 + 1)) / (cols - cols % 2)

    u_grid, v_grid = np.meshgrid(cols_scale, rows_scale)

    u_grid = np.fft.ifftshift(u_grid)
    v_grid = np.fft.ifftshift(v_grid)

    if progress_callback is not None:
        progress_callback(1,3)

    grad_x_F = np.fft.fft2(grad_x)
    grad_y_F = np.fft.fft2(grad_y)

    if progress_callback is not None:
        progress_callback(2,3)

    nominator = (-1j * u_grid * grad_x_F) + (-1j * v_grid * grad_y_F)
    denominator = (u_grid**2) + (v_grid**2) + 1e-16

    Z_F = nominator / denominator
    Z_F[0, 0] = 0.0

    Z = np.real(np.fft.ifft2(Z_F))

    if progress_callback is not None:
        progress_callback(3,3)

    return (Z - np.min(Z)) / (np.max(Z) - np.min(Z))


def apply(normals_img, seamless, progress_callback):
    """Computes a height map from the given normal map. 'normals_img' must be a numpy array
    in C,H,W format (with C as RGB). 'seamless' is a bool that should indicates if 'normals_img'
    is seamless."""

    # Flip height axis
    flip_img = np.flip(normals_img, axis=1)

    # Get gradients from normal map
    grad_x, grad_y = normals_to_grad(flip_img)
    grad_x = np.flip(grad_x, axis=0)
    grad_y = np.flip(grad_y, axis=0)

    # If non-seamless chosen, expand gradients
    if not seamless:
        grad_x, grad_y = copy_flip(grad_x, grad_y)

    # Compute height
    pred_img = frankot_chellappa(-grad_x, grad_y, progress_callback=progress_callback)
    
    # Cut to valid part if gradients were expanded
    if not seamless:
        height, width = normals_img.shape[1], normals_img.shape[2]
        pred_img = pred_img[:height, :width]

    # Expand single channel the three channels (RGB)
    return np.stack([pred_img, pred_img, pred_img])
