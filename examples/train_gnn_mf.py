import argparse
import os
import sys
import time
sys.path.append("../")
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.nmf_data_utils import SampleGenerator
from src.models.gcn import GCN_SEngine
from src.models.gnn_mf import MFEngine
from src.train_engine import TrainEngine
from src.utils.common_util import update_args
from src.utils.monitor import Monitor



def parse_args():
    """ Parse args from command line

    """
    parser = argparse.ArgumentParser(description="Run MF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/gnn_mf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
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
        "--remark", nargs="?", type=str, help="remark",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initial learning rate.")
    parser.add_argument("--reg", nargs="?", type=float, help="regularization.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


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


class MF_train(TrainEngine):
    """ An instance class from the TrainEngine base class

        """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """
        self.config = config
        super(MF_train, self).__init__(self.config)
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

        plain_adj, norm_adj, mean_adj = self.dataset.get_adj_mat()
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
        self.config["norm_adj"] = norm_adj
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
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["pre_train"] == 0:
            print(self.config["pre_train"])
            self.train_gcn()
        else:
            pass

        self.train_mf()
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.dataset.test, self.engine.model)

    def train_mf(self):
        """ Train NeuMF

        Returns:
            None
        """
        if self.config["loss"] == "bpr":
            train_loader = self.sample_generator.pairwise_negative_train_loader(
                self.config["batch_size"], self.config["device_str"]
            )
        elif self.config["loss"] == "bce":
            train_loader = self.sample_generator.uniform_negative_train_loader(
                self.config["num_negative"],
                self.config["batch_size"],
                self.config["device_str"],
            )
        else:
            raise ValueError(
                f"Unsupported loss type {self.config['loss']}, try other options: 'bpr' or 'bce'"
            )

        self.engine = MFEngine(self.config)
        self.model_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["save_name"]
        )
        self._train(self.engine, train_loader, self.model_save_dir)

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


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    train_engine = MF_train(config)
    train_engine.train()
