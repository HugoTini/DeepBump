import numpy as np
import pathlib
import onnxruntime as ort
import math

try:
    from . import utils_inference
except ImportError:
    # Cannot use . import when using as CLI
    import utils_inference

# Disable MS telemetry
ort.disable_telemetry_events()


def downscale_x2(img):
    """Downscale image by a factor of 2"""

    c, h, w = img.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Image dimensions must be even to downscale by a factor of 2.")
    reshaped_image = img.reshape(c, h // 2, 2, w // 2, 2)
    return reshaped_image.mean(axis=(2, 4))


def tiles_split(img, tile_size):
    """Returns list of tiles from the given image and the padding used to fit the tiles
    in it. Input image must have dimension C,H,W."""

    img_h, img_w = img.shape[1], img.shape[2]
    pad_h = (tile_size - (img_h % tile_size)) % tile_size
    pad_w = (tile_size - (img_w % tile_size)) % tile_size

    # Pad to be multiple of tile_size
    pad_left = pad_w // 2
    pad_right = pad_left if pad_w % 2 == 0 else pad_left + 1
    pad_top = pad_h // 2
    pad_bottom = pad_top if pad_h % 2 == 0 else pad_top + 1
    img = utils_inference.pad(img, pad_left, pad_right, pad_top, pad_bottom)
    img_h, img_w = img.shape[1], img.shape[2]

    # Split in tiles
    tiles = []
    for i in range(0, img_h, tile_size):
        for j in range(0, img_w, tile_size):
            tiles.append(img[:, i : i + tile_size, j : j + tile_size])

    return tiles, (pad_left, pad_right, pad_top, pad_bottom)


def pixel_shuffle(img, tile_size):
    """Applies smoothing to tiles edges"""

    # Columns
    for col_idx in range(tile_size, img.shape[2], tile_size):
        img = random_pair_shuffle(img, col_idx, col_idx - 1)
    # Rows
    img = np.transpose(img, (0, 2, 1))
    for col_idx in range(tile_size, img.shape[2], tile_size):
        img = random_pair_shuffle(img, col_idx, col_idx - 1)
    img = np.transpose(img, (0, 2, 1))
    return img


def random_pair_shuffle(img, col1, col2, ratio=0.5):
    """Smooth given col by averaging + pixel shuffle"""

    def get_avg(col, img):
        col_before_img = img[:, :, col - 1] if col - 1 >= 0 else img[:, :, -1]
        col_img = img[:, :, col]
        col_after_img = (
            img[:, :, col + 1] if col1 + 1 < img.shape[2] else img[:, :, col]
        )
        return (col_before_img + col_img + col_after_img) / 3.0

    # Average of both columns over their neighboor columns
    avg_col1, avg_col2 = get_avg(col1, img), get_avg(col2, img)
    img[:, :, col1], img[:, :, col2] = avg_col1, avg_col2

    # Permute some random pixels
    permute_vec = np.random.rand(img.shape[1]) < ratio
    img[:, :, col1][:, permute_vec], img[:, :, col2][:, permute_vec] = (
        img[:, :, col2][:, permute_vec],
        img[:, :, col1][:, permute_vec],
    )

    return img


def tiles_merge(tiles, tile_size, img_shape, paddings, upscale_factor=4):
    """Merges the list of tiles given an upscale factor and without overlap.
    img_size is the original size, before upscale & padding."""

    h_range = math.ceil(img_shape[1] / tile_size)
    w_range = math.ceil(img_shape[2] / tile_size)
    pad_left, pad_right, pad_top, pad_bottom = paddings
    width = img_shape[2] + pad_left + pad_right
    height = img_shape[1] + pad_top + pad_bottom

    # Upscale dims
    tile_size *= upscale_factor
    width *= upscale_factor
    height *= upscale_factor
    pad_left *= upscale_factor
    pad_right *= upscale_factor
    pad_top *= upscale_factor
    pad_bottom *= upscale_factor

    merged = np.zeros([img_shape[0], height, width])
    idx = 0
    for h in range(0, h_range):
        for w in range(0, w_range):
            h_from, h_to = h * tile_size, (h + 1) * tile_size
            w_from, w_to = w * tile_size, (w + 1) * tile_size
            merged[:, h_from:h_to, w_from:w_to] = tiles[idx]
            idx += 1

    merged = pixel_shuffle(merged, tile_size)

    return merged[:, pad_top : height - pad_bottom, pad_left : width - pad_right]


def apply(color_img, scale_factor, progress_callback):
    """Upscale image. 'color_img' must be a numpy array in C,H,W format (with C as RGB).
    'factor'' must be 'x2' or 'x4'."""

    # Remove alpha & convert to fp16 (model is in fp16)
    img = color_img[0:3].astype(np.float32)

    # Load model
    print("DeepBump Low Res -> High Res : loading model")
    addon_path = str(pathlib.Path(__file__).parent.absolute())
    ort_session = ort.InferenceSession(
        addon_path + "/upscale256.onnx", providers=["CPUExecutionProvider"]
    )

    # Split in tiles
    print("DeepBump Low Res -> High Res : generating")
    tile_size = 256
    tiles, paddings = tiles_split(img, tile_size)

    # Upscale each tile
    pred_tiles = utils_inference.tiles_infer(
        tiles, ort_session, progress_callback=progress_callback
    )

    # Merge tiles
    print("DeepBump Low Res -> High Res : merging")
    pred_img = tiles_merge(pred_tiles, tile_size, img.shape, paddings)

    # Clip to [0 .1]
    pred_img[pred_img > 1.0] = 1.0
    pred_img[pred_img < 0.0] = 0.0

    # Resize according to scale factor
    if scale_factor == "x2":
        pred_img = downscale_x2(pred_img)

    return pred_img
