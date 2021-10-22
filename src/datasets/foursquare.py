import os

import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords

from src.datasets.dataset_base import DatasetBase
from src.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)

# download_url
NYC_URL = r"http://www.google.com/url?q=http%3A%2F%2Fwww-public.it-sudparis.eu%2F~zhang_da%2Fpub%2Fdataset_ubicomp2013.zip&sa=D&sntz=1&usg=AFQjCNHNikdtb3uON4wkzuj0q_d8sdJcBw"
# pre-defined threshold
TAG_THRE = 5
WORD_THRE = 5


class Foursquare(DatasetBase):
    def __init__(self):
        """Foursquare

        Foursquare NYC dataset.
        """
        super().__init__("foursquare", url=NYC_URL)

    def preprocess(self):
        """Preprocess the raw file

        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory
        """
        file_name = os.path.join(
            self.raw_path, self.dataset_name, f"dataset_ubicomp2013_checkins.txt"
        )

        data = pd.read_table(
            file_name,
            header=None,
            engine="python",
            sep="\t",
            names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
        )
        data[DEFAULT_RATING_COL] = np.ones(len(data))
        print(data)
        self.save_dataframe_as_npz(
            data,
            os.path.join(self.processed_path, f"{self.dataset_name}_interaction.npz"),
        )

    def make_fea_vec(self):
        """ make feature vector
        1. for item (venue), we use their tags as features. 1 indicates that the venue has that tag, 0 otherwise
        2. for user, we use their tips as features, where each word is a feature.
        For both cases, we can vary threshold to filter out those features (tags or word) with low frequency.
        """

        print(f"Making user and item feature vactors for dataset {self.dataset_name}")

        cols = ["item", "tags"]
        data = pd.read_table(
            f"{self.dataset_dir}/raw/foursquare/dataset_ubicomp2013_tags.txt",
            names=cols,
            sep="\t",
            engine="python",
        )
        item_feat_li = self.item_feat(data)

        cols = ["user", "item", "tips"]
        data = pd.read_table(
            f"{self.dataset_dir}/raw/foursquare/dataset_ubicomp2013_tips.txt",
            names=cols,
            sep="\t",
            engine="python",
        )
        user_feat_li = self.user_feat(data)
        np.savez_compressed(
            f"{self.dataset_dir}/processed/feature_vec.npz",
            user_feat=user_feat_li,
            item_feat=item_feat_li,
        )
        item_featue_dic = {}
        user_featue_dic = {}
        for user_feat in user_feat_li:
            user_featue_dic[user_feat[0]] = user_feat[1:]
        for item_feat in item_feat_li:
            item_featue_dic[item_feat[0]] = item_feat[1:]
        return user_featue_dic, item_featue_dic

    def load_fea_vec(self):
        """Loading feature vectors for users and items.

        Returns:
            user_feat (numpy.ndarray): The first column is the user id, rest column are feat vectors
            item_feat (numpy.ndarray): The first column is the itm id, rest column are feat vectors

        """

        if not os.path.exists(f"{self.dataset_dir}/processed/feature_vec.npz"):
            return self.make_fea_vec()
        print(f"Loading user and item feature vectors for dataset {self.dataset_name}")
        loaded = np.load(f"{self.dataset_dir}/processed/feature_vec.npz")
        print("test")
        user_feat, item_feat = loaded["user_feat"], loaded["item_feat"]
        item_featue_dic = {}
        user_featue_dic = {}
        for user_feat in user_feat:
            user_featue_dic[user_feat[0]] = user_feat[1:]
        for item_feat in item_feat:
            item_featue_dic[item_feat[0]] = item_feat[1:]
        return user_featue_dic, item_featue_dic

    def user_feat(self, data):
        """ make user feature vector
        Args:
            data (DataFrame): DataFrame of user-item-tips
        Returns:
            user_feat (numpy.ndarray): The first column is the user id, rest column are feat vectors
        """
        string = ""
        for i in data["tips"]:
            string += i
        string_rm = remove_stopwords(string)
        list_rm = string_rm.split(" ")
        filter_tips = list(set([i for i in list_rm if list_rm.count(i) > WORD_THRE]))

        user_tips_map = {}
        list_tips = list(data["tips"])
        list_user = list(data["user"])
        unique_user = list(set(list_user))

        user_feat = np.zeros((len(unique_user), len(filter_tips) + 1), dtype=int)

        u_tip = ""
        for u in unique_user:
            tip_index = np.where(np.isin(list_user, u))[0]
            #     print(u)
            #     print(tip_index)
            #     input("Press Enter to continue...")
            for i in list(tip_index):
                u_tip += list_tips[i]
            user_tips_map[u] = u_tip
            u_tip = ""
        index = 0
        for u in unique_user:
            user_feat[index][0] = u
            user_feat[index][
                list([
                    i + 1
                    for i in np.where(np.isin(filter_tips, user_tips_map[u].split(" ")))
                ][0])
            ] = 1
            index += 1
        return user_feat

    def item_feat(self, data):
        """ make item feature vector
        Args:
            data (DataFrame): DataFrame of venue-tags
        Returns:
            item_feat (numpy.ndarray): The first column is the itm id, rest column are feat vectors
        """

        item_tag_map = {}
        for i in range(len(data)):
            item = data["item"][i]
            tags = data["tags"][i]
            if type(tags) == str:
                tags = tags.split(",")
                item_tag_map[item] = tags
            else:
                item_tag_map[item] = []
        list_tags = []
        for i in list(item_tag_map.values()):
            list_tags.extend(i)

        filter_tags_list = list(
            set([i for i in list_tags if list_tags.count(i) > TAG_THRE])
        )
        item_feat = np.zeros((len(item_tag_map), len(filter_tags_list) + 1), dtype=int)
        index = 0
        for item in item_tag_map:
            item_feat[index][0] = item
            item_feat[index][
                list([i + 1 for i in np.where(np.isin(filter_tags_list, item_tag_map[item]))][0])
            ] = 1
            index += 1
        return item_feat
