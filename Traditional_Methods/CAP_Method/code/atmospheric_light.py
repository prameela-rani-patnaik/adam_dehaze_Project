import numpy as np

def atmospheric_light(image, dark_channel):

    H, W, _ = image.shape
    imsize = H * W
    numpx = np.floor(imsize / 1000).astype(int)

    dark_channel_Vec = dark_channel.ravel()
    ImVec = image.reshape(imsize, 3)

    indices = np.argsort(dark_channel_Vec)
    indices = indices[-numpx:]

    atmSum = np.zeros(3, dtype=np.float32)

    for ind in range(numpx):
        atmSum += ImVec[indices[ind]]

    A = atmSum / numpx

    return A