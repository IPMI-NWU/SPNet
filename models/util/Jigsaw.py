import random
import numpy as np
import torch


def Jigsaw(imgs, num_x, num_y, shuffle_index=None):
    split_w, split_h = int(imgs.shape[2] / num_x), int(imgs.shape[3] / num_y)
    out_imgs = torch.zeros_like(imgs)
    imgs = imgs.unsqueeze(0)
    # -----------------------
    # 分块
    # -----------------------
    patches = torch.split(imgs, split_w, dim=3)
    patches = [torch.split(p, split_h, dim=4) for p in patches]
    patches = torch.cat([torch.cat(p, dim=0) for p in patches], dim=0)
    # -----------------------
    # shuffle_index为空则打乱, 否则还原
    # -----------------------
    if shuffle_index is None:
        shuffle_index = np.random.permutation(num_x * num_y)
    else:
        shuffle_index = list(shuffle_index)
        shuffle_index = [shuffle_index.index(i) for i in range(num_x * num_y)]
    patches = patches[shuffle_index]
    # -----------------------
    # 拼接
    # -----------------------
    x_index, y_index = 0, 0
    for patch in patches:
        out_imgs[:, :, y_index:y_index + split_h, x_index:x_index + split_w] = patch
        x_index += split_w
        if x_index == out_imgs.shape[2]:
            x_index = 0
            y_index += split_h
    return out_imgs, shuffle_index


def RandomBrightnessContrast(img, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
    output = torch.zeros_like(img)
    threshold = 0.5

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output
