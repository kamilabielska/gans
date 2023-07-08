import os
import glob
import numpy as np
import cv2

from tqdm import tqdm


def resize_images(path, size):
    print('resizing...')
    non_square = 0

    with tqdm(total=len(os.listdir(path))) as progress:
        for f in glob.glob('**/*.jpg', recursive=True, root_dir=path):
            filepath = os.path.join(path, f)
            try:
                image = cv2.imread(filepath)
                height, width, chan = image.shape
            except AttributeError:
                image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                height, width, chan = image.shape

            if height == width and height != size:
                image = cv2.resize(image, (size, size))
                cv2.imwrite(filepath, image)

            elif height != width:
                os.remove(filepath)
                non_square +=1

            progress.update(1)

    print(fr'{non_square} non-square images discarded')
    print('done')