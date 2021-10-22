import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.torch_engine import Engine
from src.utils.common_util import print_dict_as_table, timeit


class GCN_S(torch.nn.Module):
    """Initialize embedding with the single graph.

    """

    def __init__(self, config):
        super(GCN_S, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layers = (
            [int(i) for i in config["layers"].split("-")]
            if "layers" in config
            else config["gcn_config"]["layers"]
        )
        self.n_layers = len(self.layers)
        self.dropout = nn.ModuleList()
        self.u_gcn_weights = nn.ModuleList()
        self.i_gcn_weights = nn.ModuleList()
        self.layers = [self.emb_dim] + self.layers
        self.dropout_rate = config["gcn_config"]["dropout"]
        # Create GNN layers
        self.user_fea_norm_adj, self.item_fea_norm_adj = (
            config["user_fea_norm_adj"].to(self.device),
            config["item_fea_norm_adj"].to(self.device),
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

        for i in range(self.n_layers):
            self.u_gcn_weights.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            self.i_gcn_weights.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            self.dropout.append(nn.Dropout(self.dropout_rate))

        self.embedding_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        # self.global_bias.weight.data.fill_(0.0)
        init_range = 0.1 * (self.emb_dim) ** (-1 / 2)
        nn.init.uniform_(self.embedding_user.weight, -init_range, init_range)
        nn.init.uniform_(self.embedding_item.weight, -init_range, init_range)

        # nn.init.normal_(self.embedding_user.weight)
        # nn.init.normal_(self.embedding_item.weight)

    def forward(self):
        """ Perform GNN function on users and item embeddings
        Args:
            user_fea_norm_adj (torch sparse tensor): the norm adjacent matrix of the user-user similarity matrix
            item_fea_norm_adj (torch sparse tensor): the norm adjacent matrix of the item-item similarity matrix
        Returns:
            u_embeddings (tensor): processed user embeddings
            i_embeddings (tensor): processed item embeddings
        """
        u_embeddings = self.embedding_user.weight
        i_embeddings = self.embedding_item.weight

        for i in range(self.n_layers):
            u_embeddings = torch.sparse.mm(self.user_fea_norm_adj, u_embeddings)
            u_embeddings = self.u_gcn_weights[i](u_embeddings)
            u_embeddings = self.act(u_embeddings)
            u_embeddings = self.dropout[i](u_embeddings)
            u_embeddings = F.normalize(u_embeddings, p=2, dim=1)

            i_embeddings = torch.sparse.mm(self.item_fea_norm_adj, i_embeddings)
            i_embeddings = self.i_gcn_weights[i](i_embeddings)
            i_embeddings = self.act(i_embeddings)
            i_embeddings = self.dropout[i](i_embeddings)
            i_embeddings = F.normalize(i_embeddings, p=2, dim=1)

        return u_embeddings, i_embeddings

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
            scores = (
                torch.mul(ua_embeddings[users_t], ia_embeddings[items_t]).sum(dim=1)
                # + self.user_bias.weight[users]
                # + self.item_bias.weight[items]
            )
            scores += self.user_bias.weight[users].squeeze()
            scores += self.item_bias.weight[items].squeeze()
        return scores


class GCN_SEngine(Engine):
    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        self.config = config
        print_dict_as_table(config, tag="GCN config")
        self.model = GCN_S(config)
        self.regs = (
            config["regs"] if "regs" in config else config["gcn_config"]["regs"]
        )  # reg is the regularisation
        self.batch_size = config["batch_size"]
        self.num_batch = config["num_batch"]
        super(GCN_SEngine, self).__init__(config)
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

        emb_loss = self.regs * regularizer
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

        emb_loss = self.regs * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
