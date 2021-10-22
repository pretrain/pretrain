import argparse
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
sys.path.append("../")

from src.datasets.nmf_data_utils import SampleGenerator
from src.models.gcn import GCN_SEngine
from src.models.gnn_ncf import NeuMFEngine
from src.models.mlp import MLPEngine
from src.train_engine import TrainEngine
from src.utils.common_util import update_args
from src.utils.monitor import Monitor






def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run GNN-NCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/gnn_ncf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        help="Options are: 'gcn', 'mlp', 'ncf', and 'ncf_gcn'",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        help="Options are: ml-100k, ml-1m and foursquare",
    )
    parser.add_argument(
        "--data_split",
        nargs="?",
        type=str,
        help="leave_one_out",
    )
    parser.add_argument(
        "--root_dir", nargs="?", type=str, help="Working directory",
    )
    parser.add_argument(
        "--device", nargs="?", type=str, help="Device",
    )
    parser.add_argument(
        "--loss", nargs="?", type=str, help="loss: bpr or bce",
    )
    parser.add_argument(
        "--activator", nargs="?", type=str, help="activator",
    )
    parser.add_argument(
        "--remark", nargs="?", type=str, help="remark",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initial learning rate.")
    parser.add_argument("--regs", nargs="?", type=float, help="regularization.")
    parser.add_argument("--layers", nargs="?", type=str, help="layers of gcn.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


class NCF_train(TrainEngine):
    """ An instance class from the TrainEngine base class

        """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """
        self.config = config
        super(NCF_train, self).__init__(self.config)
        self.load_dataset()
        self.build_data_loader()
        self.gpu_id, self.config["device_str"] = self.get_device()

    def build_data_loader(self):
        # ToDo: Please define the directory to store the adjacent matrix
        user_fea_norm_adj, item_fea_norm_adj = self.dataset.make_fea_sim_mat()
        self.sample_generator = SampleGenerator(ratings=self.dataset.train)
        self.config["user_fea_norm_adj"] = sparse_mx_to_torch_sparse_tensor(
            user_fea_norm_adj
        )
        self.config["item_fea_norm_adj"] = sparse_mx_to_torch_sparse_tensor(
            item_fea_norm_adj
        )
        self.config["num_batch"] = self.dataset.n_train // self.config["batch_size"] + 1
        self.config["n_users"] = self.dataset.n_users
        self.config["n_items"] = self.dataset.n_items

    def _train(self, engine, train_loader, save_dir):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            if self.config["validate"]:
                self.eval_engine.train_eval(
                    self.dataset.valid[0], self.dataset.test[0], engine.model, epoch
                )
            else:
                self.eval_engine.train_eval(
                    None, self.dataset.test[0], engine.model, epoch
                )

    def train(self):
        """ Main training navigator

        Returns:

        """

        # Options are: 'gcn', 'mlp', 'ncf', and 'ncf_gcn';
        # Train NeuMF without pre-train
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["model"] == "ncf":
            self.train_ncf()
        elif self.config["model"] == "gcn":
            self.train_gcn()
        elif self.config["model"] == "mlp":
            self.train_mlp()
        elif self.config["model"] == "ncf_gcn":
            self.train_gcn()
            while self.eval_engine.n_worker:
                print(f"Wait 15s for the complete of eval_engine.n_worker")
                time.sleep(15)  # wait the
            self.train_mlp()
            while self.eval_engine.n_worker:
                print(f"Wait 15s for the complete of eval_engine.n_worker")
                time.sleep(15)  # wait the
            self.train_ncf()
        else:
            raise ValueError(
                "Model type error: Options are: 'gcn', 'mlp', 'ncf', and 'ncf_gcn'."
            )
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.dataset.test, self.engine.model)

    def train_ncf(self):
        """ Train NeuMF

        Returns:
            None
        """
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["num_negative"], self.config["batch_size"]
        )
        self.engine = NeuMFEngine(self.config)
        self.neumf_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["neumf_config"]["save_name"]
        )
        self._train(self.engine, train_loader, self.neumf_save_dir)

    def train_gcn(self):
        """ Train GCN

        Returns:
            None
        """
        train_loader = self.dataset
        # Train GCN
        self.engine = GCN_SEngine(self.config)
        self.gcn_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["gcn_config"]["save_name"]
        )
        self._train(self.engine, train_loader, self.gcn_save_dir)
        while self.eval_engine.n_worker:
            print(f"Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the

    def train_mlp(self):
        """ Train MLP

        Returns:
            None
        """
        # Train MLP
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["num_negative"], self.config["batch_size"]
        )
        self.engine = MLPEngine(self.config)
        self.mlp_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["mlp_config"]["save_name"]
        )
        self._train(self.engine, train_loader, self.mlp_save_dir)

        while self.eval_engine.n_worker:
            print(f"Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    ncf = NCF_train(config)
    ncf.train()
