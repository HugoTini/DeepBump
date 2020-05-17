from PIL import Image
import numpy as np
import infer
import onnxruntime as ort
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--img_path', dest='img_path', required=True,
                    help='path to the input image')
parser.add_argument('-n', '--normal_path', dest='normal_path', required=True,
                    help='path where the generated normal map is saved')
parser.add_argument('-o', '--overlap', dest='overlap', required=True,
                    help='tiles overlap size, must be "small", "medium" or "big"',
                    choices=['small', 'medium', 'big'])
args = parser.parse_args()

# load image
img = np.array(Image.open(args.img_path)) / 255.0
# reshape to C,H,W
img = np.transpose(img, (2, 0, 1))
# grayscale
img = np.mean(img[0:3], axis=0, keepdims=True)

# split image in tiles
print('tilling')
tile_size = (256, 256)
overlaps = {'small': 20, 'medium': 50, 'big': 124}
stride_size = (tile_size[0]-overlaps[args.overlap], tile_size[1]-overlaps[args.overlap])
tiles, paddings = infer.tiles_split(img, tile_size, stride_size)

# predict tiles normal map
print('generating')

def progress_print(current, total):
    print('{}/{}'.format(current, total))
    
ort_session = ort.InferenceSession('./deepbump256.onnx')
pred_tiles = infer.tiles_infer(tiles, ort_session, progress_callback=progress_print)

# merge tiles
print('merging')
pred_img = infer.tiles_merge(pred_tiles, stride_size, (3, img.shape[1], img.shape[2]), paddings)

# save image 
pred_img = pred_img.transpose((1,2,0))
pred_img = Image.fromarray((pred_img*255.0).astype(np.uint8))
pred_img.save(args.normal_path)  