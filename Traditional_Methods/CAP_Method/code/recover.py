import numpy as np

def recover(image, A, d, beta):

    if A.shape == (3,):
        A = A.reshape(1, 1, 3)

    H, W, C = image.shape

    t = np.exp(-beta * d)

    clip_t = np.clip(t, 0.1, 0.9)

    epsilon = 1e-8

    I_desaze = np.zeros_like(image)

    for i in range(C):
        I_desaze[:, :, i] = (
            (image[:, :, i] - A[0, 0, i]) /
            (clip_t + epsilon)
            + A[0, 0, i]
        )

    I_desaze = np.clip(I_desaze, 0, 1)

    return I_desaze, t, clip_t