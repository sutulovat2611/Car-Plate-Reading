
import numpy as np
from skimage.util import random_noise
from PIL import Image


def noise_filtering():
    print("Hello World!")


def add_noise(img):
    # Add noise to the image
    img = random_noise(img, mode = 's&p', amount = 0.3) # salt-and-pepper 
    img= random_noise(img, mode = 'gaussian', seed=None, clip=True) #gaussian
    img = random_noise(img, mode='speckle', var=0.15, clip=True) #speckle
    img = random_noise(img, mode='poisson', seed=1) #poisson

    img_to_show = Image.fromarray(np.array(255*img, dtype='uint8'))
    img_to_show.show()
    return img

if __name__ == "__main__":
    # Image pre-processing

    # Load image
    img = Image.open("images/IMG_1201.jpg")
    img = np.array(img)

    # Step 1: Image resizing & normalization

    # Step 2: Noise reduction
    # Adding noise to an image to test the noise reduction technique
    img = add_noise(img)
    noise_filtering()

    # Step 3: Image segmentation

    # Step 4: Perspective correction


