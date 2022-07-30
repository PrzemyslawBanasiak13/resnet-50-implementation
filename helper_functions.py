import matplotlib.pyplot as plt
import numpy as np
import random
import os


def plot_images(label="0", folder="human-detection-dataset", img_count=4):
    """
    Plot <img_count> random images from <folder_name> from subfolder labeled <label>
    """
    axes = []
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)

    for i in range(img_count):
        im = plt.imread(f"{folder}/{label}/{random.randint(0, len(os.listdir(f'{folder}/{label}')) - 1)}.png")
        axes.append(fig.add_subplot(1, img_count, i + 1))
        subplot_title = ("Image " + str(i))
        axes[-1].set_title(subplot_title)
        plt.imshow(im)

    fig.tight_layout()
    plt.show()
