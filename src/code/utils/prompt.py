import numpy as np


def get_bounding_box(image=None, nobox=True):

    if nobox:
        # bbox = [0, 0, image.size[0], image.size[1]]
        bbox = [np.random.randint(0, 20),
                np.random.randint(0, 20),
                image.shape[0] - np.random.randint(0, 20),
                image.shape[1] - np.random.randint(0, 20)]

    return bbox