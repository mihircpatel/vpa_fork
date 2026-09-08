#!/usr/bin/python
"""Common utilities

Replace this with a more detailed description of what this file contains.
"""
import json
import time
import pickle
import sys
import csv
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import torch

from PIL import Image

from vispr import DS_ROOT

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def load_attributes(attr_list_path=None):
    """
    Returns mappings: {attribute_id -> attribute_name} and {attribute_id -> idx}
    where attribute_id = 'aXX_YY' (string),
    attribute_name = description (string),
    idx in [0, 67] (int)
    :return:
    """
    if attr_list_path is None:
        attributes_path = osp.join(DS_ROOT, 'attributes.tsv')
    else:
        attributes_path = attr_list_path
    attr_id_to_name = dict()
    attr_id_to_idx = dict()

    with open(attributes_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = [row for row in ts if row.get('idx', '') != '']

        for row in rows:
            attr_id_to_name[row['attribute_id']] = row['description']
            attr_id_to_idx[row['attribute_id']] = int(row['idx'])

    return attr_id_to_name, attr_id_to_idx


def labels_to_vec(labels, attr_id_to_idx):
    n_labels = len(attr_id_to_idx)
    label_vec = np.zeros(n_labels)
    for attr_id in labels:
        label_vec[attr_id_to_idx[attr_id]] = 1
    return label_vec


def reload_model_weights(model, weights_path, strict=False, map_location='cpu'):
    """Load a PyTorch checkpoint into a model in a way that works across checkpoints.

    Supports checkpoints stored either as a raw state_dict or as a dict containing
    keys such as 'state_dict', 'model_state_dict', or 'model'. It also strips
    common prefixes such as 'module.' and 'model.' that appear in DataParallel or
    wrapped checkpoints.
    """
    if weights_path is None:
        return model

    checkpoint = torch.load(weights_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict')
        if state_dict is None:
            state_dict = checkpoint.get('model_state_dict')
        if state_dict is None:
            state_dict = checkpoint.get('model')
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f'Expected a state_dict-like object in {weights_path}, got {type(state_dict).__name__}')

    cleaned_state = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f'Checkpoint contains a non-string key: {type(key).__name__}')
        fixed_key = key.replace('module.', '')
        if fixed_key.startswith('model.'):
            fixed_key = fixed_key[len('model.'):]
        cleaned_state[fixed_key] = value

    return model.load_state_dict(cleaned_state, strict=strict)


def load_model_weights(model, weights_path, strict=False, map_location='cpu'):
    return reload_model_weights(model, weights_path, strict=strict, map_location=map_location)
