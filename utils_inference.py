import numpy as np


def pad(img, left, right, top, bottom):
    return np.pad(img, ((0, 0), (top, bottom), (left, right)), mode='wrap')


def tiles_split(img, tile_size, stride_size):
    '''Returns list of tiles from the given image and the padding used to fit the tiles
     in it. Input image must have dimension C,H,W.'''

    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    img_h, img_w = img.shape[1], img.shape[2]

    # stride must be even
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    # stride must be greater or equal than half tile
    assert (stride_h >= tile_h/2) and (stride_w >= tile_w/2)
    # stride must be smaller or equal tile size
    assert (stride_h <= tile_h) and (stride_w <= tile_w)

    # find total height & width padding sizes
    pad_h, pad_w = 0, 0
    remainer_h = (img_h-tile_h) % stride_h
    remainer_w = (img_w-tile_w) % stride_w
    if remainer_h != 0:
        pad_h = stride_h-remainer_h
    if remainer_w != 0:
        pad_w = stride_w-remainer_w

    # if tile bigger than image, pad image to tile size
    if tile_h > img_h:
        pad_h = tile_h-img_h
    if tile_w > img_w:
        pad_w = tile_w-img_w

    # pad image, add extra stride to padding to avoid pyramid
    # weighting leaking onto the valid part of the picture
    pad_left = pad_w//2 + stride_w
    pad_right = pad_left if pad_w % 2 == 0 else pad_left+1
    pad_top = pad_h//2 + stride_h
    pad_bottom = pad_top if pad_h % 2 == 0 else pad_top+1
    img = pad(img, pad_left, pad_right, pad_top, pad_bottom)
    img_h, img_w = img.shape[1], img.shape[2]

    # extract tiles
    h_range = ((img_h-tile_h) // stride_h) + 1
    w_range = ((img_w-tile_w) // stride_w) + 1
    tiles = np.empty([h_range*w_range, img.shape[0], tile_h, tile_w])
    idx = 0
    for h in range(0, h_range):
        for w in range(0, w_range):
            h_from, h_to = h*stride_h, h*stride_h + tile_h
            w_from, w_to = w*stride_w, w*stride_w + tile_w
            tiles[idx] = img[:, h_from:h_to, w_from:w_to]
            idx += 1

    return tiles, (pad_left, pad_right, pad_top, pad_bottom)


def tiles_infer(tiles, ort_session, progress_callback=None):
    '''Infer each tile with the given model. progress_callback will be called with 
    arguments : current tile idx and total tiles amount (used to show progress on 
    cursor in Blender).'''

    out_channels = 3  # normal map RGB channels
    tiles_nb = tiles.shape[0]
    pred_tiles = np.empty(
        (tiles_nb, out_channels, tiles.shape[2], tiles.shape[3]))

    for i in range(tiles_nb):
        if progress_callback != None:
            progress_callback(i+1, tiles_nb)
        pred_tiles[i] = ort_session.run(None,
                                        {'input': tiles[i:i+1].astype(np.float32)})[0]

    return pred_tiles


def generate_mask(tile_size, stride_size):
    '''Generates a pyramidal-like mask. Used for mixing overlapping predicted tiles.'''

    tile_h, tile_w = tile_size
    stride_h, stride_w = stride_size
    ramp_h = tile_h - stride_h
    ramp_w = tile_w - stride_w

    mask = np.ones((tile_h, tile_w))

    # ramps in width direction
    mask[ramp_h:-ramp_h, :ramp_w] = np.linspace(0, 1, num=ramp_w)
    mask[ramp_h:-ramp_h, -ramp_w:] = np.linspace(1, 0, num=ramp_w)
    # ramps in height direction
    mask[:ramp_h, ramp_w:-ramp_w] = np.transpose(np.linspace(0, 1, num=ramp_h)[None],
                                                 (1, 0))
    mask[-ramp_h:, ramp_w:-ramp_w] = np.transpose(np.linspace(1, 0, num=ramp_h)[None],
                                                  (1, 0))

    # Assume tiles are squared
    assert ramp_h == ramp_w
    # top left corner
    corner = np.rot90(corner_mask(ramp_h), 2)
    mask[:ramp_h, :ramp_w] = corner
    # top right corner
    corner = np.flip(corner, 1)
    mask[:ramp_h, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 0)
    mask[-ramp_h:, -ramp_w:] = corner
    # bottom right corner
    corner = np.flip(corner, 1)
    mask[-ramp_h:, :ramp_w] = corner

    return mask


def corner_mask(side_length):
    '''Generates the corner part of the pyramidal-like mask. 
    Currently, only for square shapes.'''

    corner = np.zeros([side_length, side_length])

    for h in range(0, side_length):
        for w in range(0, side_length):
            if h >= w:
                sh = h / (side_length-1)
                corner[h, w] = 1-sh
            if h <= w:
                sw = w / (side_length-1)
                corner[h, w] = 1-sw

    return corner-0.25*scaling_mask(side_length)


def scaling_mask(side_length):

    scaling = np.zeros([side_length, side_length])

    for h in range(0, side_length):
        for w in range(0, side_length):
            sh = h / (side_length-1)
            sw = w / (side_length-1)
            if h >= w and h <= side_length-w:
                scaling[h, w] = sw
            if h <= w and h <= side_length-w:
                scaling[h, w] = sh
            if h >= w and h >= side_length-w:
                scaling[h, w] = 1-sh
            if h <= w and h >= side_length-w:
                scaling[h, w] = 1-sw

    return 2*scaling


def tiles_merge(tiles, stride_size, img_size, paddings):
    '''Merges the list of tiles into one image. img_size is the original size, before 
    padding.'''

    _, tile_h, tile_w = tiles[0].shape
    pad_left, pad_right, pad_top, pad_bottom = paddings
    height = img_size[1] + pad_top + pad_bottom
    width = img_size[2] + pad_left + pad_right
    stride_h, stride_w = stride_size

    # stride must be even
    assert (stride_h % 2 == 0) and (stride_w % 2 == 0)
    # stride must be greater or equal than half tile
    assert (stride_h >= tile_h/2) and (stride_w >= tile_w/2)
    # stride must be smaller or equal tile size
    assert (stride_h <= tile_h) and (stride_w <= tile_w)

    merged = np.zeros((img_size[0], height, width))
    mask = generate_mask((tile_h, tile_w), stride_size)

    h_range = ((height-tile_h) // stride_h) + 1
    w_range = ((width-tile_w) // stride_w) + 1

    idx = 0
    for h in range(0, h_range):
        for w in range(0, w_range):
            h_from, h_to = h*stride_h, h*stride_h + tile_h
            w_from, w_to = w*stride_w, w*stride_w + tile_w
            merged[:, h_from:h_to, w_from:w_to] += tiles[idx]*mask
            idx += 1

    return merged[:, pad_top:-pad_bottom, pad_left:-pad_right]


def normalize(img):
    'Normalize each pixel to unit vector.'

    img = img-0.5
    img = img / np.sqrt(np.sum(img*img, axis=0, keepdims=True))
    return (img*0.5)+0.5
