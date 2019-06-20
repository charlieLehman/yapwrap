from PIL import Image, ImageOps, ImageFilter
import numpy as np

class RandomNegative(object):
    """Take the negative of an image randomly.
    Args:
        probability (float): Probability that a negative will be used.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if np.random.rand() <= self.probability:
            return ImageOps.invert(img)
        else:
            return img
