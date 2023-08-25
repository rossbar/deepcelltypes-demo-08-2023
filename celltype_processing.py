import os
from tqdm import tqdm
import numpy as np
import scipy as sp
import tifffile as tff
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.measure import regionprops
from tensorflow.keras.models import load_model

def histogram_normalization(image, kernel_size=None):
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.
    Args:
        image (numpy.array): numpy array of phase image data with shape
            (H, W, C). Note there is no batch index here.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    image = image.astype("float32")
    assert len(image.shape) == 3

    for channel in range(image.shape[-1]):
        X = image[..., channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            image[..., channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[..., channel] = X
    return image


def pad_cell(X: np.ndarray, y: np.ndarray,  crop_size: int):
    delta = crop_size // 2
    X = np.pad(X, ((delta, delta), (delta, delta), (0,0)))
    y = np.pad(y, ((delta, delta), (delta, delta)))
    return X, y


def get_crop_box(centroid, delta):
    minr = int(centroid[0]) - delta
    maxr = int(centroid[0]) + delta
    minc = int(centroid[1]) - delta
    maxc = int(centroid[1]) + delta
    return np.array([minr, minc, maxr, maxc])


def get_neighbor_masks(mask, cbox, cell_idx):
    """Returns binary masks of a cell and its neighbors. This function expects padding around
    the edges, and will throw an error if you hit a wrap around."""
    minr, minc, maxr, maxc = cbox
    assert np.issubdtype(mask.dtype, np.integer) and isinstance(cell_idx, int)

    cell_view = mask[minr:maxr, minc:maxc]
    binmask_cell = (cell_view == cell_idx).astype(np.int32)

    binmask_neighbors = (cell_view != cell_idx).astype(np.int32) * (cell_view != 0).astype(
        np.int32
    )
    return np.stack([binmask_cell, binmask_neighbors])


def preprocess(orig_img, mask, ch_names):
    import tensorflow as tf
    from pathlib import Path
    import yaml

    # Get model channels and cell types
    model_dir = Path("model/saved_model")
    config_path = Path("model/config.yaml")
    with open(config_path, "r") as fh:
        model_config = yaml.load(fh, yaml.Loader)
    master_channel_lst = model_config["channels"]
    master_cell_types = np.asarray(model_config["cell_types"])

    master_channels = {ch.upper(): ch for ch in master_channel_lst}
    channel_lst, img = [], []
    for idx, ch in enumerate(ch_names):
        key = ch.upper()
        if key in master_channels:
            channel_lst.append(master_channels[key])
            img.append(orig_img[..., idx].T)
    multiplex_img = np.asarray(img)

    marker_intersection = set(master_channels) & {ch.upper() for ch in ch_names}

    assert len(marker_intersection) == multiplex_img.shape[0]
    print(f"Used marker panel: {marker_info['intersection']}")

    kernel_size = 128
    crop_size = 64 
    num_channels = 32 # minimum of all dataset channel lengths
    # check master list against channel_lst
    assert not set(channel_lst) - set(master_channel_lst)
    assert len(channel_lst) == class_X.shape[-1]


    X = histogram_normalization(class_X, kernel_size=kernel_size)

    # B, Y, X, C
    X, y = pad_cell(X, mask, crop_size)
    return X, y


def convert_to_model_input(X, y):
    props = regionprops(y, cache=False)
    appearances_list = []
    padding_mask_lst = []
    channel_names_lst = []
    label_lst = []
    real_len_lst = []
    crop_size = 64
    rs = 32
    num_channels = 32 # minimum of all dataset channel lengths

    for prop_idx, prop in tqdm(enumerate(props)):
        curr_cell = prop_idx + 1
        label = prop.label
        delta = crop_size // 2
        cbox = get_crop_box(prop.centroid, delta)
        neighbor = get_neighbor_masks(y, cbox, prop.label)

        minr, minc, maxr, maxc = cbox
        raw_patch = X[minr:maxr, minc:maxc, :]
        raw_patch = raw_patch.transpose((2, 0, 1))[..., None]  # (C, H, W, 1)
        raw_patch = tf.image.resize(raw_patch, (rs, rs))
        neighbor = tf.image.resize(neighbor[..., None], (rs, rs))[..., 0]

        padding_len = num_channels - raw_patch.shape[0]
        neighbor = tf.reshape(
            neighbor,
            (*neighbor.shape, 1),
        )
        neighbor = tf.transpose(neighbor, [3, 1, 2, 0])

        neighbor = tf.tile(neighbor, [tf.shape(raw_patch)[0], 1, 1, 1])
        image_aug_neighbor = tf.concat(
            [raw_patch, tf.cast(neighbor, dtype=tf.float32)], axis=-1
        )

        paddings_mask = tf.constant([[0, 1], [0, 0], [0, 0], [0, 0]])
        paddings = paddings_mask * padding_len


        appearances = tf.pad(
            image_aug_neighbor, paddings, "CONSTANT", constant_values=-1.0
        )

        channel_names = tf.concat(
            [channel_lst, tf.repeat([b"None"], repeats=padding_len)], axis=0
        )

        mask_vec = tf.concat(
            [
                tf.repeat([True], repeats=raw_patch.shape[0]),
                tf.repeat([False], repeats=padding_len),
            ],
            axis=0,
        )

        mask_vec = tf.cast(mask_vec, tf.float32)
        m1, m2 = tf.meshgrid(mask_vec, mask_vec)
        padding_mask = m1 * m2

        # append each of these to list, conver to tensor
        appearances_list.append(appearances)
        padding_mask_lst.append(padding_mask)
        channel_names_lst.append(channel_names)
        label_lst.append(label)
        real_len_lst.append(raw_patch.shape[0])

    appearances_list = tf.convert_to_tensor(appearances_list)
    padding_mask_lst = tf.convert_to_tensor(padding_mask_lst)
    channel_names_lst = tf.convert_to_tensor(channel_names_lst)
    label_lst = tf.convert_to_tensor(label_lst)
    real_len_lst = tf.convert_to_tensor(real_len_lst)

    model_input = {
        "appearances": appearances_list,
        "channel_padding_masks": padding_mask_lst,
        "channel_names": channel_names_lst,
        "cell_idx_label": label_lst,
        "real_len": real_len_lst,
        "inpaint_channel_name": tf.convert_to_tensor(
            ["None"] * appearances_list.shape[0]
        ),
    }
    
    return model_input

def run_model(model_input, model_dir, output_categories):

    ctm = load_model(model_dir, compile=False)

    logits = ctm.predict(model_input)[0]

    pred_idx = np.argmax(sp.special.softmax(logits, axis=1), axis=1)
    cell_type_predictions = output_categories[1:][pred_idx]

    return dict(enumerate(cell_type_predictions))

