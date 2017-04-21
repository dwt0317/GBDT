# -*- coding:utf-8 -*-
from math import log
from random import sample


class Tree:
    def __init__(self):
        self.split_feature = None  #分裂特征
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = True
        self.conditionValue = None  #分裂值
        self.leafNode = None

    def get_predict_value(self, instance):   #predict_value即在训练过程中计算出的各叶子结点的Gama_jm
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    #返回树的信息
    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info

    def get_attributes(self):
        if not self.leftTree or not self.rightTree:
            return None
        leftAttrs= self.leftTree.get_attributes()
        rightAttrs = self.rightTree.get_attributes()
        attributes = [self.split_feature,]
        attributes.extend(leftAttrs)
        attributes.extend(rightAttrs)
        return attributes       


class LeafNode:
    def __init__(self, idset):
        self.idset = idset  #样本id（行数）
        self.predictValue = None

    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_ternimal_regions(targets, self.idset)


#计算的是各样本的梯度波动程度
def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error

#用于多分类
def FriedmanMSE(left_values, right_values):
    """
    参考Friedman的论文Greedy Function Approximation: A Gradient Boosting Machine中公式35
    """
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))

def reachLeaf(remainedSet,dataset, targets, loss, leaf_nodes):
    node = LeafNode(remainedSet)
    node.update_predict_value(targets, loss)
    leaf_nodes.append(node)
    tree = Tree()
    tree.leafNode = node
    return tree

#remainedSet 分裂后剩余的样本; split points: 用于训练的特征个数？  targets: 梯度/残差  split_points改成其他值的话，每次迭代的特征集合会变成随机的
def construct_decision_tree(dataset, remainedSet, targets, depth, leaf_nodes, max_depth, loss, remainAttr, criterion='MSE', split_points=0):
    if depth < max_depth and len(remainAttr) > 0 :
        # todo 通过修改这里可以实现选择多少特征训练
        attributes=remainAttr
        mse = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        for attribute in attributes:  #feature的名称
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            if is_real_type and split_points > 0 and len(attrValues) > split_points:   
                attrValues = sample(attrValues, split_points)
            # 选择分裂feature以及其value
            for attrValue in attrValues:  
                leftIdSet = []
                rightIdSet = []
                for Id in remainedSet:  
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    # 将满足条件的放入左子树
                    if (is_real_type and value < attrValue)or(not is_real_type and value == attrValue):
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sum_mse = MSE(leftTargets)+MSE(rightTargets)
                if mse < 0 or sum_mse < mse:  #如果分裂后的误差更小
                    selectedAttribute = attribute  #分裂feature
                    conditionValue = attrValue   #分裂值
                    mse = sum_mse
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
        if not selectedAttribute or mse < 0 or len(selectedLeftIdSet) < 1  or len(selectedRightIdSet)<1: #叶子结点
            # print mse, selectedAttribute
            # raise ValueError("cannot determine the split attribute.")  #一旦执行了raise语句，raise后面的语句将不能执行
            return reachLeaf(remainedSet,dataset, targets, loss, leaf_nodes)
        tree = Tree()
        tree.split_feature = selectedAttribute
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        attributes.remove(selectedAttribute)  #可以选择有放回或无放回的选择特征
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth+1, leaf_nodes, max_depth, loss, attributes)
        tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, depth+1, leaf_nodes, max_depth, loss, attributes)
        return tree
    else:  # 是叶子节点
        return reachLeaf(remainedSet,dataset,targets,loss, leaf_nodes)



