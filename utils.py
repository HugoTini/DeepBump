import numpy as np

def bl_image_to_np(bl_img):
    """Converts a Blender image to a numpy C,H,W numpy array."""

    # Convert to C,H,W numpy array
    width = bl_img.size[0]
    height = bl_img.size[1]
    channels = bl_img.channels
    np_img = np.array(bl_img.pixels)
    np_img = np.reshape(np_img, (channels, width, height), order='F')
    np_img = np.transpose(np_img, (0, 2, 1))

    # Flip height
    np_img = np.flip(np_img, axis=1)

    # Remove alpha
    np_img = np_img[0:3]

    return np_img

def np_to_bl_pixels(np_img):
    """Converts a C,H,W numpy image array to an array of pixel suited for 
    blender internal images pixels."""

    # Flip height
    img = np.flip(np_img, axis=1)
    # Add alpha channel
    height, width = np_img.shape[1], np_img.shape[2]
    img = np.concatenate(
        [img, np.ones((1, height, width))], axis=0)
    # Flatten to array
    pixels = np.transpose(img, (0, 2, 1)).flatten('F')
    
    return pixels