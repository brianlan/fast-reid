# encoding: utf-8
"""
@author:  rongyi.lan
@contact: rongyi.lan@clobotics.com
"""

import glob
import os.path as osp
import re
import warnings
import logging

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Retail(ImageDataset):
    """Retail Dataset - contains products from CEA, CCTH and P&G.

    Dataset statistics:
        - Number of SKUs: 673 (train), 574 (query), 1661 (gallery), no SKU overlapping.
        - Number of images: 35635 (train) + 11319 (query) + 13310 (gallery).
    """
    dataset_name = "retail"

    def __init__(self, root, dataset_indices, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))

        # allow alternative directory structure
        train_indices_path = dataset_indices.get("TRAIN")
        query_indices_path = dataset_indices.get("QUERY")
        gallery_indices_path = dataset_indices.get("GALLERY")
        label_map_path = dataset_indices.get("LABEL_MAP")

        self.check_before_run([
            train_indices_path,
            query_indices_path,
            gallery_indices_path,
            label_map_path,
        ])

        train = self._read_indices(train_indices_path)
        query = self._read_indices(query_indices_path, label_map_path=label_map_path, is_train=False)
        gallery = self._read_indices(gallery_indices_path, label_map_path=label_map_path, init_camid=1000000, is_train=False)

        super(Retail, self).__init__(train, query, gallery, **kwargs)

    def _read_indices(self, path, label_map_path=None, init_camid=0, is_train=True):
        with open(path, "r") as f:
            data = [l.strip().split() for l in f]
        all_labels = sorted({d[1] for d in data})
        if label_map_path is None:
            label2id = {label: _id for _id, label in enumerate(all_labels)}
        else:
            with open(label_map_path, "r") as f:
                label2id = {c.strip(): i for i, c in enumerate(f)}

        # pretend all the images are taken from different cameras to make the code work.
        if is_train:
            return [(d[0], f"{self.dataset_name}_{label2id[d[1]]}", init_camid + i) for i, d in enumerate(data)]
        else:
            return [(d[0], label2id[d[1]], init_camid + i) for i, d in enumerate(data)]
