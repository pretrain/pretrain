import argparse
import sys
import os
from tqdm import tqdm
import json
import numpy as np
import torch
import time
sys.path.append("../")
from src.train_engine import TrainEngine
from src.utils.common_util import update_args
from src.utils.monitor import Monitor
from src.datasets.nmf_data_utils import SampleGenerator
from src.models.compgcn import CompGCNEngine


def parse_args():
    """ Parse args from command line
        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run COMPGCN..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/compgcn_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--tune", nargs="?", type=str, default=True, help="Tun parameter",
    )
    parser.add_argument(
        "--keep_pro", nargs="?", type=float, help="dropout", default=0.6
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")

    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--reg", nargs="?", type=float, help="regularsation."
    )
    parser.add_argument(
        "--dataset", nargs="?", type=str, help="dataset name"
    )
    parser.add_argument(
        "--base_dim", nargs="?", type=int, help="base dimension"
    )
    parser.add_argument(
        "--n_base", nargs="?", type=int, help="number of base"
    )

    parser.add_argument(
        "--late_dim", nargs="?", type=int, help="latent dimension"
    )
    parser.add_argument(
        "--device", nargs="?", type=str, help="device"
    )
    parser.add_argument(
        "--loss", nargs="?", type=str, help="loss"
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


class COMPGCN_train(TrainEngine):
    """ An instance class from the TrainEngine base class
    """

    def __init__(self, config):
        """Constructor
        Args:
            config (dict): All the parameters for the model
        """

        self.config = config
        super(COMPGCN_train, self).__init__(config)
        self.load_dataset()
        self.build_data_loader()
        self.gpu_id, self.config["device_str"] = self.get_device()
    def build_data_loader(self):
        (
            user_edge_list,
            user_edge_type,
            item_edge_list,
            item_edge_type,
            self.config["n_user_fea"],
            self.config["n_item_fea"],
        ) = self.dataset.make_multi_graph()
        self.sample_generator = SampleGenerator(ratings=self.dataset.train)
        self.config["user_edge_list"] = torch.LongTensor(user_edge_list)
        self.config["user_edge_type"] = torch.LongTensor(user_edge_type)
        self.config["item_edge_list"] = torch.LongTensor(item_edge_list)
        self.config["item_edge_type"] = torch.LongTensor(item_edge_type)
        self.config["num_batch"] = self.dataset.n_train // self.config["batch_size"] + 1
        self.config["n_users"] = self.dataset.n_users
        self.config["n_items"] = self.dataset.n_items
        plain_adj, norm_adj, mean_adj = self.dataset.get_adj_mat()
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
        self.config["norm_adj"] = norm_adj

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
        self.train_compgcn()
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.dataset.test, self.engine.model)
        return self.eval_engine.best_valid_performance

    def train_compgcn(self):
        """ Train compgcn

        Returns:
            None
        """
        train_loader = self.dataset
        # Train RGCN
        self.engine = CompGCNEngine(self.config)
        self.model_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["compgcn_config"]["save_name"]
        )
        self._train(self.engine, train_loader, self.model_save_dir)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)
    update_args(config, args)
    compgcn = COMPGCN_train(config)
    compgcn.train()
