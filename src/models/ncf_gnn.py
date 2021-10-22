import os

import torch

from src.models.gcn import GCN_S
from src.models.mlp_gnn import MLP
from src.models.torch_engine import Engine
from src.utils.common_util import print_dict_as_table


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = config["n_users"]
        self.num_items = config["n_items"]
        self.latent_dim_gcn = config["emb_dim"]
        self.latent_dim_mlp = config["emb_dim"]
        self.layers = [config["emb_dim"] * 2] + config["layers"]

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim_gcn
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim_gcn
        )

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(
            in_features=config["layers"][-1] + self.latent_dim_gcn, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def predict(self, user_indices, item_indices):
        user_indices = torch.LongTensor(user_indices).to(self.device)
        item_indices = torch.LongTensor(item_indices).to(self.device)
        with torch.no_grad():
            return self.forward(user_indices, item_indices)

    def init_weight(self):
        pass


class NeuMFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config, gcn_config=None, mlp_config=None):
        self.model = NeuMF(config)
        print_dict_as_table(config, tag="Neumf config")
        self.gcn_config = gcn_config
        self.mlp_config = mlp_config
        super(NeuMFEngine, self).__init__(config)
        print(self.model)
        if gcn_config is not None and mlp_config is not None:
            self.load_pretrain_weights()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, "model"), "Please specify the exact model !"
        users, items, ratings = (
            users.to(self.device),
            items.to(self.device),
            ratings.to(self.device),
        )
        self.optimizer.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""
        gcn_model = GCN_S(self.gcn_config)
        self.resume_checkpoint(
            os.path.join(self.config["model_save_dir"], self.config["pretrain_gcn"]),
            gcn_model,
        )
        self.model.embedding_user_mf.weight.data = gcn_model.user_embedding.weight.data
        self.model.embedding_item_mf.weight.data = gcn_model.item_embedding.weight.data

        mlp_model = MLP(self.mlp_config)
        self.resume_checkpoint(
            os.path.join(self.config["model_save_dir"], self.config["pretrain_mlp"]),
            mlp_model,
        )
        self.model.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.model.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
