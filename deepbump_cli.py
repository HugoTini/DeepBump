from PIL import Image
import numpy as np
import infer
import onnxruntime as ort
from argparse import ArgumentParser


def progress_print(current, total):
    print('{}/{}'.format(current, total))


# Disable MS telemetry
ort.disable_telemetry_events()

# Parse cli arguments
parser = ArgumentParser()
parser.add_argument('-i', '--img_path', dest='img_path', required=True,
                    help='path to the input image')
parser.add_argument('-n', '--normal_path', dest='normal_path', required=True,
                    help='path where the generated normal map is saved')
parser.add_argument('-o', '--overlap', dest='overlap', required=True,
                    help='tiles overlap size, must be "small", "medium" or "large"',
                    choices=['small', 'medium', 'large'])
args = parser.parse_args()

# Load image
img = np.array(Image.open(args.img_path)) / 255.0
# Reshape to C,H,W
img = np.transpose(img, (2, 0, 1))
# Grayscale
img = np.mean(img[0:3], axis=0, keepdims=True)

# Split image in tiles
print('tilling')
tile_size = 256
overlaps = {'small':  tile_size//6, 'medium':  tile_size//4,
            'large': tile_size//2}
stride_size = tile_size-overlaps[args.overlap]
tiles, paddings = infer.tiles_split(img, (tile_size, tile_size),
                                    (stride_size, stride_size))

# Predict tiles normal map
print('generating')
ort_session = ort.InferenceSession('./deepbump256.onnx')
pred_tiles = infer.tiles_infer(tiles, ort_session,
                               progress_callback=progress_print)

# Merge tiles
print('merging')
pred_img = infer.tiles_merge(pred_tiles, (stride_size, stride_size),
                             (3, img.shape[1], img.shape[2]),
                             paddings)

# Save image
pred_img = pred_img.transpose((1, 2, 0))
pred_img = Image.fromarray((pred_img*255.0).astype(np.uint8))
pred_img.save(args.normal_path)
