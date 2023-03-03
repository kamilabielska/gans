import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def plot_history(fit, metrics, val=True):
    fig, axes = plt.subplots(1, len(metrics), figsize=(15,4))

    for i, which in enumerate(metrics):
        ax = axes[i]
        ax.plot(fit.history[which], label=which)
        if val:
            ax.plot(fit.history['val_'+which], label='val_'+which)
        ax.set_xlabel('epoch')
        ax.set_ylabel(which)
        ax.legend()


def plot_sample_images(dataset, n=4, rescaled=False):
    fig, axes = plt.subplots(1, n, figsize=(14,4))
    axes = axes.flatten()
    images = dataset.take(1).unbatch().shuffle(1000).as_numpy_iterator()
    i = 0
    while i < n:
        image = next(images).squeeze()
        image = image if not rescaled else (image + 1) / 2
        cmap = 'gray' if np.ndim(image) == 2 else None
        axes[i].imshow(image, cmap=cmap)
        axes[i].axis('off')
        i += 1


def gif_from_image_folder(folder_path, gif_path, duration=0.2):
    filenames = sorted(os.listdir(folder_path), key=lambda x: int((x.split('.'))[0]))
    images = [imageio.imread(os.path.join(folder_path, x)) for x in filenames]
    imageio.mimsave(gif_path, images, duration=duration)