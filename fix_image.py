from PIL import Image

import os


def fix_size():
    # Initializing the desired dimensions
    desired_w=27
    desired_h=48
    fill_color=(0, 0, 0, 255) # black color to fill empty space

    # Specifying directory
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'character_image/1/1_03.jpg')

    # Opening image
    im = Image.open(file_path)

    # Determining the size of the image
    x, y = im.size
    
    # Getting the desired ratio
    desired_ratio = desired_w / desired_h

    # Resizing the image
    w = max(desired_w, x)
    h = int(w / desired_ratio)
    if h < y:
        h = y
        w = int(h * desired_ratio)

    # Filling the empty space with black color
    new_im = Image.new('RGBA', (w, h), fill_color)
    new_im.paste(im, ((w - x) // 2, (h - y) // 2))
    new_im.resize((desired_w, desired_h))
    return new_im




   