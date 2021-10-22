import sys

from src.datasets.data_split import filter_user_item
from src.datasets.movielens import Movielens_1m, Movielens_100k

sys.path.append("../")



if __name__ == "__main__":
    dataset = Movielens_100k()
    # dataset.preprocess()
    # user_feat, item_feat = dataset.load_fea_vec()
    # print(user_feat.shape, item_feat.shape)
    # interactions = dataset.load_interaction()
    # interactions = filter_user_item(interactions, 1, 20)
    dataset.make_leave_one_out()
    #
    # dataset = Movielens_100k()
    # dataset.preprocess()
    # interactions = dataset.load_interaction()
    # interactions = filter_user_item(interactions, 1, 20)
    # dataset.make_leave_one_out(interactions)
