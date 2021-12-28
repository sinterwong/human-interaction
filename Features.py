from os import name
from typing import SupportsComplex
import numpy as np


class FeaturesManager(object):
    def __init__(self, features: np.ndarray = None, names: list = None, dims: int = 128, threshold: float = 0.9) -> None:
        super().__init__()
        self.dims = dims
        self.__features = None
        self.__feature_mapping = None
        self.count = 0
        self.threshold = threshold
        self.init()
        if features is not None:
            self.mult_insert(features, names)

    def __len__(self):
        return self.__features.shape[0] - 1

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return super().__str__()

    def __cosine_similar(self, v):
        if len(v.shape) == 1:
            v = np.expand_dims(v, axis=0)
        assert self.dims == v.shape[1], "The feature length must equal dims."
        num = np.dot(self.__features, v.T)
        denom = np.linalg.norm(self.__features, axis=1).reshape(-1, 1) \
            * np.linalg.norm(v, axis=1).reshape(-1, 1)
        res = num / denom
        res[np.isneginf(res)] = 0
        return res

    @property
    def shape(self):
        return self.__features.shape

    @property
    def num_people(self):
        return self.count - 1

    @property
    def peoples(self):
        return list(self.__feature_mapping.keys())[1:]

    def init(self):
        self.__feature_mapping = {"unknown": 0}
        self.__features = np.ones([1, self.dims], dtype=np.float32)
        self.count += 1

    def mult_insert(self, features: np.ndarray, names: list = None):
        insert_num = 0
        if not names:
            names = dict(
                zip(range(1, features.shape[0] + 1), range(1, features.shape[0] + 1)))
        for f, n in zip(features, names):
            insert_num += self.single_insert(f, n)
        return insert_num

    def single_insert(self, feature, name):
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)

        assert self.dims == feature.shape[1], "The feature length must equal dims."
        if name in self.__feature_mapping.keys():
            # 替换之前的特征
            self.__features[self.__feature_mapping[name]] = feature
            return 0
        else:
            # 向后插入特征
            self.__features = np.concatenate(
                [self.__features, feature], axis=0)
            self.__feature_mapping[name] = self.count
            self.count += 1
            return 1

    def delete_feature(self, name):
        self.__features = np.delete(
            self.__features, self.__feature_mapping[name], 0)
        del self.__feature_mapping[name]

    def identify(self, feature):
        peoples = list(self.__feature_mapping.keys())
        scores = self.__cosine_similar(feature)
        res_idx = np.argmax(scores, axis=0)[0]
        if scores[res_idx] < self.threshold:
            return peoples[res_idx], scores[res_idx]
        return peoples[res_idx], float(scores[res_idx])
