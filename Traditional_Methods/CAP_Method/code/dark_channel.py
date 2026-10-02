import numpy as np

def dark_channel(image):
    H, W, _ = image.shape

    patch_size = 15
    pad_size = patch_size // 2

    dc = np.zeros((H, W), dtype=np.float32)

    imJ = np.pad(
        image,
        ((pad_size, pad_size),
         (pad_size, pad_size),
         (0, 0)),
        mode='constant',
        constant_values=np.inf
    )

    for j in range(H):
        for i in range(W):
            patch = imJ[
                j:j + patch_size,
                i:i + patch_size,
                :
            ]

            dc[j, i] = np.min(patch)

    return dc