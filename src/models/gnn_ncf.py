import os

import torch
import torch.nn as nn

from src.models.gcn import GCN_S
from src.models.mlp import MLP
from src.models.torch_engine import Engine
from src.utils.common_util import timeit


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.n_layers = config["mlp_config"]["n_layers"]
        self.dropout = config["dropout"]
        self.latent_dim_mlp = self.emb_dim * (2 ** (self.n_layers)) // 2
        self.latent_dim_gcn = self.emb_dim

        self.embedding_user_mlp = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_item_mlp = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_mlp
        )
        self.embedding_user_mf = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim_gcn
        )
        self.embedding_item_mf = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim_gcn
        )
        self.user_bias = torch.nn.Embedding(self.n_users, 1)
        self.item_bias = torch.nn.Embedding(self.n_items, 1)

        MLP_modules = []
        for i in range(self.n_layers):
            input_size = self.emb_dim * (2 ** (self.n_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*MLP_modules)
        self.affine_output = torch.nn.Linear(
            in_features=self.emb_dim * 2 + 2, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        user_bias = self.user_bias(user_indices)
        item_bias = self.item_bias(item_indices)

        mlp_vector = torch.cat(
            [user_embedding_mlp, item_embedding_mlp], dim=-1
        )  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = torch.cat([mf_vector, user_bias, item_bias], dim=-1)
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
    """Engine for training & evaluating GCN model"""

    def __init__(self, config):
        self.config = config
        self.model = NeuMF(config)
        self.loss = torch.nn.BCELoss()
        super(NeuMFEngine, self).__init__(config)
        #print(self.model)
        #if self.config["model"] == "ncf_gcn":
        #    self.load_pretrain_weights()
        #else:
        #    self.init_weights()

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

    @timeit
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

    def init_weights(self):
        nn.init.normal_(self.model.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.model.embedding_item_mf.weight, std=0.01)
        nn.init.normal_(self.model.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.model.embedding_user_mlp.weight, std=0.01)
        for m1 in self.model.fc_layers:
            if isinstance(m1, nn.Linear):
                nn.init.xavier_uniform_(m1.weight)
        nn.init.kaiming_uniform_(
            self.model.affine_output.weight, a=1, nonlinearity="sigmoid"
        )

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GCN model"""
        # load GCN_S model
        print("loading pretrain weight")
        gcn_model = GCN_S(self.config)
        if self.config["pre_train"] == 0:
            gcn_save_dir = os.path.join(
                self.config["model_save_dir"], self.config["gcn_config"]["save_name"]
            )
        else:
            gcn_save_dir = self.config["root_dir"] + "/pre_train_weight/" + "gcn.model_" + self.config["dataset"]

        self.resume_checkpoint(
            gcn_save_dir, gcn_model,
        )
        self.model.embedding_user_mf.weight.data = gcn_model.embedding_user.weight.data
        self.model.embedding_item_mf.weight.data = gcn_model.embedding_item.weight.data
        self.model.user_bias.weight.data = gcn_model.user_bias.weight.data
        self.model.item_bias.weight.data = gcn_model.item_bias.weight.data

        # load MLP model
        mlp_model = MLP(self.config)
        mlp_save_dir = os.path.join(
            self.config["model_save_dir"], self.config["mlp_config"]["save_name"]
        )
        self.resume_checkpoint(
            mlp_save_dir, mlp_model,
        )
        self.model.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.model.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for (m1, m2) in zip(self.model.fc_layers, mlp_model.fc_layers):
            if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                m1.weight.data.copy_(m2.weight)
                m1.bias.data.copy_(m2.bias)
