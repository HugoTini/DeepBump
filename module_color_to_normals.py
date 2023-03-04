import numpy as np
import pathlib
import onnxruntime as ort
try :
    from . import utils_inference
except ImportError:
    # Cannot use . import when using as CLI
    import utils_inference

# Disable MS telemetry
ort.disable_telemetry_events()

def apply(color_img, overlap, progress_callback):
    """Computes a normal map from the given color map. 'color_img' must be a numpy array
    in C,H,W format (with C as RGB). 'overlap' must be one of 'SMALL', 'MEDIUM', 'LARGE'."""

    # Remove alpha & convert to grayscale
    img = np.mean(color_img[0:3], axis=0, keepdims=True)

    # Split image in tiles
    print("DeepBump Color → Normals : tilling")
    tile_size = 256
    overlaps = {
        "SMALL": tile_size // 6,
        "MEDIUM": tile_size // 4,
        "LARGE": tile_size // 2,
    }
    stride_size = tile_size - overlaps[overlap]
    tiles, paddings = utils_inference.tiles_split(
        img, (tile_size, tile_size), (stride_size, stride_size)
    )

    # Load model
    print("DeepBump Color → Normals : loading model")
    addon_path = str(pathlib.Path(__file__).parent.absolute())
    ort_session = ort.InferenceSession(addon_path + "/deepbump256.onnx")

    # Predict normal map for each tile
    print("DeepBump Color → Normals : generating")
    pred_tiles = utils_inference.tiles_infer(
        tiles, ort_session, progress_callback=progress_callback
    )

    # Merge tiles
    print("DeepBump Color → Normals : merging")
    pred_img = utils_inference.tiles_merge(
        pred_tiles,
        (stride_size, stride_size),
        (3, img.shape[1], img.shape[2]),
        paddings,
    )

    # Normalize each pixel to unit vector
    pred_img = utils_inference.normalize(pred_img)

    return pred_img
