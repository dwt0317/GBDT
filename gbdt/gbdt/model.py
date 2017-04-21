# -*- coding:utf-8 -*-
from datetime import datetime
import abc
from random import sample
from math import exp, log
from gbdt.tree import construct_decision_tree




# 回归和分类问题使用不同的损失函数
class ClassificationLossFunction(object):
    _metaclass_=abc.ABCMeta
    """分类损失函数的基类"""
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, f):
        """计算残差"""   #其实是梯度

    @abc.abstractmethod
    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, f, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值"""


class BinomialDeviance(ClassificationLossFunction):  
    """二元分类的损失函数"""
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        super(BinomialDeviance, self).__init__(1)

    def compute_residual(self, dataset, subset, f):  #每个样本都可以算出一个梯度
        residual = {}
        for id in subset:
            y_i = dataset.get_instance(id)['label']
            residual[id] = 2.0*y_i/(1+exp(2*y_i*f[id]))
        return residual

    def update_f_value(self, f, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            f[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, f, dataset):
        ids = dataset.get_instances_idset()
        for id in ids:
            f[id] = 0.0

    def update_ternimal_regions(self, targets, idset):  #到达叶子节点，计算预测值
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(2-abs(targets[id])) for id in idset])   #abs 绝对值
        return sum1 / sum2



class GBDT:
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.split_points = split_points
        self.loss = None
        self.trees = dict()

    # train data 为训练样本的id_set
    def fit(self, dataset, train_data):       
        self.loss = BinomialDeviance(n_classes=dataset.get_label_size())
        f = dict()  # 记录F_{m-1}的值
        self.loss.initialize(f, dataset)
        for iter in range(1, self.max_iter+1):  #每次迭代采样不同的样本
            subset = train_data
            if 0 < self.sample_rate < 1:
                subset = sample(subset, int(len(subset)*self.sample_rate))
            # 用损失函数的负梯度作为回归问题提升树的残差近似值
            residual = self.loss.compute_residual(dataset, subset, f)
            leaf_nodes = []
            targets = residual
            attributes=list(dataset.get_attributes())   #tuple的元素不能删除，使用list
            tree = construct_decision_tree(dataset, subset, targets, 0, leaf_nodes, self.max_depth, self.loss, attributes)
            self.trees[iter] = tree
            self.loss.update_f_value(f, tree, leaf_nodes, subset, dataset, self.learn_rate)
            train_loss = self.compute_loss(dataset, train_data, f)
            print("iter%d : train loss=%f" % (iter,train_loss))


    def compute_loss(self, dataset, subset, f):
        loss = 0.0
        for id in dataset.get_instances_idset():
            y_i = dataset.get_instance(id)['label']
            f_value = f[id]
            p_1 = 1/(1+exp(-2*f_value))
            try:
                loss -= ((1+y_i)*log(p_1)/2) + ((1-y_i)*log(1-p_1)/2)
            except ValueError as e:
                print(y_i, p_1)

        return loss/dataset.size()

    def compute_instance_f_value(self, instance):     #Fm= sum(Fi),Fi为单个样本在每颗树上的对应叶子结点的predict value
        """计算样本的f值"""
        f_value = 0.0
        for iter in self.trees:
            f_value += self.learn_rate * self.trees[iter].get_predict_value(instance)   #f值为每次迭代生成树的预测值的加权和
        return f_value

    def predict(self, instance):
        """
        对于回归和二元分类返回f值
        对于多元分类返回每一类的f值
        """
        return self.compute_instance_f_value(instance)

    def predict_prob(self, instance):
        f_value = self.compute_instance_f_value(instance)
        probs = dict()
        probs['+1'] = 1/(1+exp(-2*f_value))
        probs['-1'] = 1 - probs['+1']
        return probs

    def predict_label(self, instance):
        """预测标签"""
        probs = self.predict_prob(instance)
        predict_label = 1 if probs[1] >= probs[-1] else -1
        return predict_label

