import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import OrderedDict
import numpy as np

def remove_duplicates_and_convert_npy(val_list):
    tmp_dist = OrderedDict()
    for ele in val_list:
        tmp_dist[ele[0]] = ele[1:]
    val_list = []
    for k, v in tmp_dist.items():
        val_list.append(((k,) + v))
    ret_npy = np.zeros((len(val_list), len(val_list[0])), dtype=np.float32)
    for i, ele_tuple in enumerate(val_list):
        for j in range(len(ele_tuple)):
            ret_npy[i, j] = float(ele_tuple[j])
    return ret_npy


def temporal_smoothing(training_statistics, stride=10, window_size=100):
    """We always assume the first axis in statistics is the iteration

    Parameters
    ----------
    training_statistics
    stride
    window_size

    Returns
    -------
    smoothed_mean:
    smoothed_std:
    """
    slice_obj = slice(window_size-1, None, stride)
    iter_slice = training_statistics[slice_obj, 0:1]
    rolling_frame = pd.DataFrame(training_statistics[:, 1:]).rolling(window=window_size, center=False)
    smoothed_mean = rolling_frame.mean().as_matrix()[slice_obj, :]
    smoothed_std = rolling_frame.std().as_matrix()[slice_obj, :]
    smoothed_mean = np.concatenate([iter_slice, smoothed_mean], axis=1)
    smoothed_std = np.concatenate([iter_slice, smoothed_std], axis=1)
    return smoothed_mean, smoothed_std


def parse_log(file_path, regex):
    """

    Parameters
    ----------
    file_path
    regex

    Returns
    -------

    """
    with open(file_path) as f:
        content = f.read()
    ret = re.findall(regex, content)
    ret = remove_duplicates_and_convert_npy(ret)
    return ret
