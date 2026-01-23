import numpy as np


def conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


def relu(feature_map):
    return np.maximum(0, feature_map)


def maxpool(feature_map, size=2, stride=2):
    h, w = feature_map.shape
    pooled_h = (h - size) // stride + 1
    pooled_w = (w - size) // stride + 1

    pooled = np.zeros((pooled_h, pooled_w))

    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            pooled[i // stride, j // stride] = np.max(
                feature_map[i:i+size, j:j+size]
            )

    return pooled


def flatten(feature_maps):
    return feature_maps.flatten()

#Example
img = np.random.rand(8, 8)
kernel = np.array([ [1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1] ])

if __name__ == "__main__":
    conv = conv2d(img, kernel)
    act = relu(conv)
    pool = maxpool(act)

    print("Conv shape:", conv.shape)
    print("Pool shape:", pool.shape)
    print("Flatten length:", flatten(pool).shape)