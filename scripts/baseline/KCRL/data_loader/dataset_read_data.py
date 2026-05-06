#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read KCRL datasets from existing processed files."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from pytz import timezone
from sklearn.preprocessing import StandardScaler


SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.dag_io import load_graph_npy


class DataGenerator(object):
    def __init__(self, file_path, solution_path=None, normalize_flag=False, transpose_flag=False):
        from helpers.dir_utils import create_dir
        from helpers.log_helper import LogHelper

        loaded = np.load(file_path, allow_pickle=file_path.endswith("data.npy"))
        if file_path.endswith("data.npy"):
            loaded = loaded.item().get("x")
        self.inputdata = loaded
        self.datasize, self.d = self.inputdata.shape

        data_dir = "dataset/{}".format(
            datetime.now(timezone("Asia/Hong_Kong")).strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        )
        create_dir(data_dir)

        LogHelper.setup(log_path="{}/training.log".format(data_dir), level_str="INFO")

        _logger = logging.getLogger(__name__)
        _logger.info(print("Input data is \n", self.inputdata))

        if normalize_flag:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)
            _logger.info(print("In the NORMALIZE block \n"))
            _logger.info(print("After normalizing \n", self.inputdata))

        if solution_path is None:
            gtrue = np.zeros(self.d)
        else:
            gtrue = load_graph_npy(solution_path, transpose=transpose_flag)
            if transpose_flag:
                _logger.info(print("After transposing \n", gtrue))

        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3)
        _logger.info(print("True DAG absolutes values \n", self.true_graph))

    def gen_instance_graph(self, max_length, dimension, test_mode=False):
        seq = np.random.randint(self.datasize, size=(dimension))
        input_ = self.inputdata[seq]
        return input_.T

    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            input_ = self.gen_instance_graph(max_length, dimension)
            input_batch.append(input_)

        return input_batch
