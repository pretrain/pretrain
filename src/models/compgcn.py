import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from src.models.compgcn_conv import CompGCNConv, CompGCNConvBasis
from src.models.torch_engine import Engine
from src.utils.common_util import print_dict_as_table, timeit


class CompGCNBase(torch.nn.Module):
    def __init__(self, config):
        super(CompGCNBase, self).__init__()
        self.device = config["device_str"]
        self.n_users = config["n_users"]  # Number of users
        self.n_items = config["n_items"]  # Number of items
        self.n_user_rel = config["n_user_fea"]  # Number of relations
        self.n_item_rel = config["n_item_fea"]  # Number of relations
        self.n_bases = config["n_bases"]  # Number of bases
        self.base_dim = config["emb_dim"]   # Dimension of bases
        self.late_dim = config["late_dim"]  # Dimension of the first gcn layer
        self.emb_dim = config["emb_dim"]  # Dimension of embeddings
        self.dropout_rate = config["dropout"]  # Dimension of embeddings
        self.opn = config["opn"]
        # Create GNN layers
        (
            self.user_edge_list,
            self.user_edge_type,
            self.item_edge_list,
            self.item_edge_type,
        ) = (
            config["user_edge_list"].to(self.device),
            config["user_edge_type"].to(self.device),
            config["item_edge_list"].to(self.device),
            config["item_edge_type"].to(self.device),
        )
        if config["activator"] == "tanh":
            self.act = torch.tanh
        elif config["activator"] == "sigmoid":
            self.act = torch.sigmoid
        elif config["activator"] == "relu":
            self.act = F.relu
        elif config["activator"] == "lrelu":
            self.act = F.leaky_relu
        elif config["activator"] == "prelu":
            self.act = F.prelu
        else:
            self.act = lambda x: x

        self.embedding_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = Parameter(torch.zeros(1))
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        self.global_bias.data.fill_(0.0)
        init_range = 0.1 * (self.emb_dim) ** (-1 / 2)
        nn.init.uniform_(self.embedding_user.weight, -init_range, init_range)
        nn.init.uniform_(self.embedding_item.weight, -init_range, init_range)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.u_conv1 = CompGCNConvBasis(
            self.base_dim,
            self.late_dim,
            self.n_user_rel,
            self.n_bases,
            opn=self.opn,
            dropout=self.dropout_rate,
            act=self.act,
            device=self.device,
        )
        self.u_conv2 = CompGCNConv(
            self.late_dim,
            self.emb_dim,
            self.n_user_rel,
            opn=self.opn,
            dropout=self.dropout_rate,
            act=self.act,
            device=self.device,
        )
        self.i_conv1 = CompGCNConvBasis(
            self.base_dim,
            self.late_dim,
            self.n_item_rel,
            self.n_bases,
            opn=self.opn,
            dropout=self.dropout_rate,
            act=self.act,
            device=self.device,
        )
        self.i_conv2 = CompGCNConv(
            self.late_dim,
            self.emb_dim,
            self.n_item_rel,
            opn=self.opn,
            dropout=self.dropout_rate,
            act=self.act,
            device=self.device,
        )
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        self.global_bias.data.fill_(0.0)

    def forward(self):
        user_embed, user_rel_embed = self.u_conv1(
            self.embedding_user.weight, self.user_edge_list, self.user_edge_type
        )
        user_embed = self.dropout(user_embed)
        user_embed, user_rel_embed = self.u_conv2(
            user_embed,
            self.user_edge_list,
            self.user_edge_type,
            rel_embed=user_rel_embed,
        )

        item_embed, item_rel_embed = self.i_conv1(
            self.embedding_item.weight, self.item_edge_list, self.item_edge_type
        )
        item_embed = self.dropout(item_embed)
        item_embed, item_rel_embed = self.i_conv2(
            item_embed,
            self.item_edge_list,
            self.item_edge_type,
            rel_embed=item_rel_embed,
        )
        return user_embed, item_embed

    def predict(self, users, items):
        """ Model prediction: dot product of users and items embeddings
        Args:
            users (int):  user id
            items (int):  item id
        Return:
            scores (int): dot product
        """
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.forward()
            scores = torch.sigmoid(
                torch.sum(
                    torch.mul(ua_embeddings[users_t], ia_embeddings[items_t]).squeeze(),
                    dim=1,
                )
                + self.user_bias(users_t).squeeze()
                + self.item_bias(items_t).squeeze()
                + self.global_bias
            )
        return scores


class CompGCNEngine(Engine):
    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        self.config = config
        print_dict_as_table(config, tag="CompGCN config")
        self.model = CompGCNBase(config)
        self.batch_size = config["batch_size"]
        self.num_batch = config["num_batch"]
        self.reg = config["reg"]
        super(CompGCNEngine, self).__init__(config)
        self.model.to(self.device)
        if "loss" in self.config:
            self.loss = self.bce_loss if self.config["loss"] == "bce" else self.bpr_loss
            print(f"using {self.config['loss']} loss...")
        else:
            self.loss = self.bpr_loss

    def train_single_batch(self, batch_data):
        """
        Args:
            batch_data (list): batch users, positive items and negative items
        Return:
            loss (float): batch loss
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        ua_embeddings, ia_embeddings = self.model.forward()

        batch_users, pos_items, neg_items = batch_data

        u_g_embeddings = ua_embeddings[batch_users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        u_bias = self.model.user_bias.weight[batch_users]
        pos_i_bias = self.model.item_bias.weight[pos_items]
        neg_i_bias = self.model.item_bias.weight[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.loss(
            u_g_embeddings,
            pos_i_g_embeddings,
            neg_i_g_embeddings,
            u_bias,
            pos_i_bias,
            neg_i_bias,
        )

        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """ Generate batch data for each batch
        Args:
            epoch_id (int):
            user (list)
            pos_i (list):
            neg_i (list):
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        n_batch = self.num_batch
        for idx in range(n_batch):
            batch_data = train_loader.sample(self.batch_size)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def bpr_loss(
        self, users, pos_items, neg_items, u_bias, pos_i_bias, neg_i_bias,
    ):
        # Calculate Binary Cross Entropy loss
        pos_scores = torch.sigmoid(
            torch.sum(torch.mul(users, pos_items), dim=1)
            + u_bias.squeeze()
            + pos_i_bias.squeeze()
        )
        neg_scores = torch.sigmoid(
            torch.sum(torch.mul(users, neg_items), dim=1)
            + u_bias.squeeze()
            + neg_i_bias.squeeze()
        )

        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def bce_loss(
        self, users, pos_items, neg_items, u_bias, pos_i_bias, neg_i_bias,
    ):
        # Calculate Binary Cross Entropy loss
        pos_scores = torch.sigmoid(
            torch.sum(torch.mul(users, pos_items), dim=1) + u_bias + pos_i_bias
        )
        neg_scores = torch.sigmoid(
            torch.sum(torch.mul(users, neg_items), dim=1) + u_bias + neg_i_bias
        )
        pos_ratings = torch.ones_like(pos_scores)
        neg_ratings = torch.zeros_like(neg_scores)
        loss = torch.nn.BCELoss()
        mf_loss = loss(pos_scores, pos_ratings) + loss(neg_scores, neg_ratings)
        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        emb_loss = self.reg * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
