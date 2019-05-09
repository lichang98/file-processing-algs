# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:29:36 2019

@author: 李畅
"""
from os import path, listdir, remove, mkdir, rename
import gensim
import numpy as np
from multiprocessing import Pool, Manager
import math
import pickle
import copy


# import io
# import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,'utf-8')


class TreeNode:
    """
    cart 树结点
    """

    def __init__(self, file_list, split_ind=None, split_val=None):
        """
        @param file_list: 属于该结点的文件
        @param split_ind: 当前结点分类的依据index
        @param split_val: 当前结点分类依据的值
        """
        self.id = None
        self.file_list = file_list
        self.split_ind = split_ind
        self.split_val = split_val
        self.type_ind = None  # 当前结点的类别（当前结点为leafnode时）
        self.left = None
        self.right = None
        self.gini_val = float('inf')
        pass

    def __str__(self) -> str:
        return 'TreeNode:[id=' + str(self.id) + ', split_ind=' + str(self.split_ind) + ', split_val=' + str(
            self.split_val) + ', type_ind=' + str(self.type_ind) + ', gini_val=' + str(self.gini_val) + ']'


def sub_process(low: int, high: int, file_list, file_vec, type_names, file_type, queue):
    """
    子进程，处理low~high 内特征的最优划分
    :param low:
    :param high:
    :param file_list:
    :param file_vec:
    :param type_names:
    :param file_type:
    :param queue:
    :return:
    """
    best_split_ind = 0
    best_split_val = 0.
    best_gini = float('inf')  # gini 指数最小值
    print('sub process solve vec dim range :' + str(low) + '-' + str(high) + ', file count' + str(len(file_list)))
    for feat_ind in range(low, high):
        # 获取当前特征的所有可能的取值
        all_possible_val = set()
        for file in file_list:
            all_possible_val |= {file_vec[file][feat_ind]}
        print('subprocess solve feat_ind:' + str(feat_ind) + ', allpossible value count=' + str(len(all_possible_val)))
        # 计算每个取值划分下的gini 指数
        for split_val in all_possible_val:
            left = []
            right = []
            for file in file_list:
                if file_vec[file][feat_ind] < split_val:
                    left.append(file)
                else:
                    right.append(file)
            # 计算当前划分下的gini 指数
            type_count_left = [0] * len(type_names)
            type_count_right = [0] * len(type_names)
            for file in left:
                type_count_left[file_type[file]] += 1
            for file in right:
                type_count_right[file_type[file]] += 1
            gini_left = 1.
            for i in range(len(type_names)):
                gini_left -= math.pow(type_count_left[i] * 1.0 / (len(left) if len(left) != 0 else (len(left) + 1)), 2)
            gini_right = 1.
            for i in range(len(type_names)):
                gini_right -= math.pow(
                    type_count_right[i] * 1.0 / (len(right) if len(right) != 0 else (len(right) + 1)), 2)
            # 计算当前的gini 系数
            gini = len(left) * 1.0 / len(file_list) * gini_left + len(right) * 1.0 / len(file_list) * gini_right
            if gini < best_gini:
                best_split_ind = feat_ind
                best_split_val = split_val
                best_gini = gini
    # 保存当前进程处理的最优划分方案
    queue.put((best_split_ind, best_split_val, best_gini))
    pass


class CART:

    def __init__(self, train_dir, stopwords_file, vecdim=300, window_sz=10,
                 max_depth=120):
        """
        @param train_dir: 训练文件所在的文件夹，文件夹下每一个类别一个目录
        @param stopwords_file: 包含停用词的文件
        @param vecdim: 词向量的维度
        @param window_sz: 词向量采样窗口的大小
        @param max_depth: cart 分类树最大深度限制, 在使用剪枝策略时，该值应设为inf
        """
        self.train_dir = train_dir
        self.stopwords_file = stopwords_file
        self.vecdim = vecdim
        self.window_sz = window_sz
        self.max_depth = max_depth
        if not path.exists('cart_tree.model'):
            if not path.exists('cart_word2vec.model'):
                with open('corpus_all_content.data', 'w', encoding='utf-8') as fw:
                    for subdir in listdir(self.train_dir):
                        for file in listdir(path.join(self.train_dir, subdir)):
                            with open(path.join(self.train_dir, subdir, file), 'r', encoding='utf-8') as fr:
                                print(fr.read(), file=fw)
                self.sentences = gensim.models.word2vec.Text8Corpus('corpus_all_content.data')
                self.model = gensim.models.word2vec.Word2Vec(self.sentences, size=self.vecdim)
                self.model.save('cart_word2vec.model')
                remove('corpus_all_content.data')
            else:
                self.model = gensim.models.word2vec.Word2Vec.load('cart_word2vec.model')
            # 计算每个文件的文件向量
            # 去除停用词，每个文件的文件向量使用该文件中所有词语的向量的均值作为该文件的文件向量
            with open(self.stopwords_file, 'r', encoding='utf-8') as fr:
                stop_word_list = fr.read().split()
            stop_word_list = [item.strip() for item in stop_word_list if item.strip()]
            word_vec = {}  # {词语：词向量, ......} 不包含停用词
            for word in self.model.wv.index2word:
                if word not in stop_word_list:
                    word_vec.update({word: self.model[word]})
            self.type_names = []  # 每个类别的名称
            for subdir in listdir(self.train_dir):
                self.type_names.append(subdir)
            self.file_type = {}  # {文件名: 所属类别ind}
            self.file_vec = {}  # {文件名称：文件向量, ...}
            for subdir in listdir(self.train_dir):
                for file in listdir(path.join(self.train_dir, subdir)):
                    self.file_type.update({file: self.type_names.index(subdir)})
                    # 读取文件, 获取所有的词向量
                    file_word_vec = np.zeros(self.vecdim, dtype=np.float32)
                    with open(path.join(self.train_dir, subdir, file), 'r', encoding='utf-8') as fr:
                        file_all_words = fr.read().split()
                    file_all_words = [item.strip() for item in file_all_words if item.strip() and item.strip()
                                      not in stop_word_list]
                    count = 0
                    for word in file_all_words:
                        if word in word_vec.keys():
                            file_word_vec += np.asarray(word_vec[word])
                            count += 1
                    file_word_vec = file_word_vec / count
                    self.file_vec.update({file: file_word_vec})
            print('train file preprocess finish...')
            self.cart_root = TreeNode(self.file_type.keys())
            self.id = 0  # 结点编号
            self._iter_generate(self.cart_root, depth=0)  # 训练生成分类树
        pass

    def _iter_generate(self, cart_node: TreeNode, depth: int):
        """
        递归生成cart 分类树
        @param cart_node: 当前处理的结点
        @param depth: 当前深度
        """
        cart_node.id = self.id
        self.id += 1
        # 判断当前结点内的文件是否属于同一类
        types = set()
        type_count = [0] * len(self.type_names)
        for file in cart_node.file_list:
            types |= {self.file_type[file]}
            type_count[self.file_type[file]] += 1
        cart_node.type_ind = np.argmax(np.asarray(type_count))
        if len(types) == 1:
            print('depth=' + str(depth) + ', current node all file in same type...')
            cart_node.type_ind = self.file_type[cart_node.file_list[0]]
            del cart_node.file_list
            cart_node.file_list = None
            return
        # 使用四进程计算最优划分点
        part_size = int(self.vecdim / 4)
        range_low = 0
        split_methods = []
        best_splits = Manager().Queue()
        process_pool = Pool(4)
        print('start subprocesses.....')
        for i in range(4):
            if i < 3:
                process_pool.apply_async(sub_process,
                                         args=(range_low, range_low + part_size, cart_node.file_list, self.file_vec,
                                               self.type_names, self.file_type, best_splits,))
            else:
                process_pool.apply_async(sub_process, args=(range_low, self.vecdim, cart_node.file_list, self.file_vec,
                                                            self.type_names, self.file_type, best_splits,))
            range_low += part_size
        process_pool.close()
        process_pool.join()
        print('getting split methods from sub processes...')
        for i in range(4):
            split_methods.append(best_splits.get())
        best_split_ind = 0
        best_split_val = 0.
        best_gini = float('inf')
        for i in range(4):
            if split_methods[i][2] < best_gini:
                best_split_ind, best_split_val, best_gini = split_methods[i]
        print('best split ind=' + str(best_split_ind) + ', best split_val=' + str(
            best_split_val) + ', lowest gini=' + str(best_gini))
        # 使用最优划分方案对当前文件划分
        left_files = []
        right_files = []
        for file in cart_node.file_list:
            if self.file_vec[file][best_split_ind] < best_split_val:
                left_files.append(file)
            else:
                right_files.append(file)
        if not left_files or not right_files or depth >= self.max_depth:
            # 当前没有可以划分的特征， leafnode
            print('no feat to split or reach max depth, current node\'s file=' +
                  str(len(cart_node.file_list)))
            type_count = [0] * len(self.type_names)
            for file in cart_node.file_list:
                type_count[self.file_type[file]] += 1
            type_ind = np.argmax(np.asarray(type_count))
            print('current node\'s type index=' + str(type_ind))
            cart_node.type_ind = type_ind
            del cart_node.file_list
            cart_node.file_list = None
            del left_files
            del right_files
        else:
            # 递归处理左右部分
            del cart_node.file_list
            cart_node.file_list = None
            cart_node.split_ind = best_split_ind  # 设置当前结点的最优分割特征
            cart_node.split_val = best_split_val  # 设置当前结点的最优分割特征的值
            cart_node.gini_val = best_gini  # 保存当前结点的最佳gini值
            cart_left = TreeNode(left_files)
            cart_node.left = cart_left
            print('curent depth:' + str(depth) + ', left_file_len=' + str(len(cart_left.file_list)))
            self._iter_generate(cart_left, depth + 1)
            cart_right = TreeNode(right_files)
            cart_node.right = cart_right
            print('current depth:' + str(depth) + ', rihgt_file_len=' + str(len(cart_right.file_list)))
            self._iter_generate(cart_right, depth + 1)
        pass

    def file_classify(self, test_dir):
        """
        对测试文件夹下的文件进行分类
        """
        test_file_list = listdir(test_dir)
        for type_name in self.type_names:
            mkdir(path.join(test_dir, type_name), 0o755)
        # 使用四个进程处理
        print('prediction start....')
        process_pool = Pool(4)
        part_size = int(len(test_file_list) / 4)
        range_low = 0
        for i in range(4):
            if i == 3:
                process_pool.apply_async(self._sub_process_predict, args=(test_file_list[range_low:], test_dir,))
            else:
                process_pool.apply_async(self._sub_process_predict,
                                         args=(test_file_list[range_low:range_low + part_size], test_dir,))
            range_low += part_size
        process_pool.close()
        process_pool.join()
        print('prediction finish.....')
        pass

    def _sub_process_predict(self, file_list, test_dir):
        """
        对测试目录下的文件进行分类
        @param file_list: 当前子进程处理的文件列表
        """
        with open(self.stopwords_file, 'r', encoding='utf-8') as fr:
            stop_word_list = fr.read().split()
        stop_word_list = [item.strip() for item in stop_word_list if item.strip()]
        for file in file_list:
            with open(path.join(test_dir, file), 'r', encoding='utf-8') as fr:
                file_words_list = fr.read().split()
            file_words_list = [item.strip() for item in file_words_list if
                               item.strip() and item.strip() not in stop_word_list]
            # 读取词向量模型
            word_vec = {}  # {词语：词向量,..}
            for word in self.model.wv.index2word:
                if word not in stop_word_list:
                    word_vec.update({word: self.model[word]})
            # 计算当前文档的向量
            file_vec = np.zeros(self.vecdim, dtype=np.float32)
            count = 0
            for word in file_words_list:
                if word in word_vec.keys():
                    file_vec += np.asarray(word_vec[word])
                    count += 1
            file_vec = file_vec * 1.0 / count
            # 根据训练得到的模型对该文件进行分类
            cart_node = self.cart_root
            while cart_node.left or cart_node.right:  # 直到结点为leafnode
                if cart_node.left and file_vec[cart_node.split_ind] < cart_node.split_val:
                    cart_node = cart_node.left
                elif cart_node.right and file_vec[cart_node.split_ind] >= cart_node.split_val:
                    cart_node = cart_node.right
                else:
                    break
            # 根据当前分类移动到对应的文件夹下
            rename(path.join(test_dir, file), path.join(test_dir, self.type_names[cart_node.type_ind], file))
        pass

    def _gen_new_tree(self, raw_tree: TreeNode, cut_node_id: int):
        """
        生成新的cart决策树
        :param raw_tree:
        :param cut_node_id: 被剪枝的结点的Id
        :return: 剪枝后的cart决策树根结点
        """
        from queue import Queue
        node_queue = Queue()
        new_queue = Queue()
        new_tree = copy.deepcopy(raw_tree)
        node_queue.put(raw_tree)
        new_queue.put(new_tree)
        nodes_count = 0
        while not node_queue.empty():
            p = node_queue.get()
            q = new_queue.get()
            nodes_count += 1
            if p.id != cut_node_id:
                if p.left:
                    node_queue.put(p.left)
                    q.left = copy.deepcopy(p.left)
                    new_queue.put(q.left)
                if p.right:
                    node_queue.put(p.right)
                    q.right = copy.deepcopy(p.right)
                    new_queue.put(q.right)
            else:
                q.left = None
                q.right = None
                q.gini_val = float('inf')
        print('new generate tree has ' + str(nodes_count) + ' nodes.')
        return new_tree

    def _clean_tree(self, cart_root: TreeNode):
        """
        清空cart 树的空间
        :param cart_root:
        :return:
        """
        if not cart_root:
            return
        if cart_root.left:
            self._clean_tree(cart_root.left)
        if cart_root.right:
            self._clean_tree(cart_root.right)
        del cart_root

    def _validate_subprocess(self, cart_node: TreeNode, validate_file_list: list,validate_file_vec ,ans_queue):
        """
        验证过程的子进程
        :param cart_node:
        :param validate_file_list:  子进程处理的文件的列表
        :param ans_queue: 将处理得到的文件名称以及对应的类别ind 保存到queue中
        :return:
        """
        print('getting validate file\'s types....')
        file_type = {}
        for file in validate_file_list:
            node = cart_node
            while node.left or node.right:
                if node.left and validate_file_vec[file][node.split_ind] < node.split_val:
                    node = node.left
                elif node.right and validate_file_vec[file][node.split_ind] >= node.split_val:
                    node = node.right
                else:
                    break
            file_type.update({file: node.type_ind})
        ans_queue.put(file_type)
        print('getting file\'s types finish....')

    def _validate_carttree(self, cart_node: TreeNode, type_count: list, validate_file_type: dict, validate_dir):
        """
        在验证集上得到cart决策树的准确率
        :param cart_node:
        :param type_count:
        :param validate_file_type:
        :param validate_dir: 验证集目录
        :return:
        """
        validate_file_list = list(validate_file_type.keys())
        # 将验证集的文档使用word2vec转换
        if not path.exists('validate_word2vec.model'):
            with open('validate_all_content.data', 'w', encoding='utf-8') as fw:
                for subdir in listdir(validate_dir):
                    for file in listdir(path.join(validate_dir, subdir)):
                        with open(path.join(validate_dir, subdir, file), 'r', encoding='utf-8') as fr:
                            print(fr.read(), file=fw)
            validate_sentence = gensim.models.word2vec.Text8Corpus('validate_all_content.data')
            validate_model = gensim.models.word2vec.Word2Vec(validate_sentence, size=self.vecdim)
            validate_model.save('validate_word2vec.model')
        else:
            validate_model = gensim.models.word2vec.Word2Vec.load('validate_word2vec.model')
        # 获取验证集每个文档的文档向量
        with open(self.stopwords_file, 'r', encoding='utf-8') as fr:
            stop_word_list = fr.read().split()
        stop_word_list = [item.strip() for item in stop_word_list if item.strip()]
        word_vec = {}  # {词语：词向量,.....}
        for word in validate_model.wv.index2word:
            if word not in stop_word_list:
                word_vec.update({word: validate_model[word]})
        file_vec = {}  # {文件名: 文件对应的向量,....}
        for subdir in listdir(validate_dir):
            for file in listdir(path.join(validate_dir, subdir)):
                with open(path.join(validate_dir, subdir, file), 'r', encoding='utf-8') as fr:
                    file_word_list = fr.read().split()
                file_word_list = [item.strip() for item in file_word_list if item.strip()]
                vec = np.zeros(self.vecdim, dtype=np.float32)  # 当前文档的向量
                count = 0
                for word in file_word_list:
                    if word in word_vec.keys():
                        vec += np.asarray(word_vec[word])
                        count += 1
                vec = vec * 1.0 / count
                file_vec.update({file: vec})  # 保存当前文档以及对应的向量
        print('validateing process start, validate file list len=' + str(len(validate_file_list)))
        part_size = int(len(validate_file_list) // 4)
        ans_queue = Manager().Queue()
        range_low = 0
        process_pool = Pool(4)
        for i in range(4):
            if i == 3:
                process_pool.apply_async(self._validate_subprocess,
                                         args=(cart_node, validate_file_list[range_low:], file_vec,ans_queue,))
            else:
                process_pool.apply_async(self._validate_subprocess,
                                         args=(cart_node, validate_file_list[range_low:range_low + part_size], file_vec,ans_queue,))
            range_low += part_size
        process_pool.close()
        process_pool.join()
        print('sub processes of validating all finish....')
        # 获取在验证集上的分类结果
        ans_validate_type = {}
        for i in range(4):
            ans_validate_type = dict(ans_validate_type, **ans_queue.get())
        # 计算分类的准确率
        print('getted classify answers on validate data set....')
        validate_type_count = [0] * len(self.type_names)
        for file, type_ind in ans_validate_type.items():
            if validate_file_type[file] == type_ind:
                validate_type_count[type_ind] += 1
        precision = 0
        for i in range(len(self.type_names)):
            precision += validate_type_count[i] * 1.0 / type_count[i] if validate_type_count[i] >0 else float('-inf')
        precision = precision * 1.0 / len(self.type_names)  # 取平均值
        return precision

    def cut_branch(self, validate_dir):
        """
        cart 树剪枝
        迭代计算生成的cart树的每个结点的gini 指数，选择使得g=(Ct-CTi)/(\Ti\-1)最小的结点，剪枝，使得其为子结点
        保存剪枝过程中所有的树，在一个独立的验证集上选择分类效果最好的
        :param validate_dir: 验证集所在的文件夹，用于选择最优的决策树，文件夹下包含以类别名命名的子文件夹，子文件夹下包含属于该类的文档
        :return:
        """
        from queue import Queue
        print('starting cut branch operations....')
        validate_file_type = {}  # {验证文件名: 所属类别ind,....}
        type_count = [0] * len(self.type_names)  # 属于每个类别的验证样本的个数
        print('getting validating files....')
        for subdir in listdir(validate_dir):
            for file in listdir(path.join(validate_dir, subdir)):
                validate_file_type.update({file: self.type_names.index(subdir)})
                type_count[self.type_names.index(subdir)] += 1
        # 每次选择gini 值最小的剪枝
        gen_tree_roots = []  # 通过剪枝生成的一系列cart决策树的根结点
        tree_root = copy.deepcopy(self.cart_root)
        # 迭代，直至根节点以及叶结点的情况
        print('generate all candidate cart trees.....')
        while tree_root.left and tree_root.right:
            node_que = Queue()
            node_que.put(tree_root)
            min_gini_val = float('inf')
            min_gini_id = None
            while not node_que.empty():
                p = node_que.get()
                if p.gini_val < min_gini_val:
                    min_gini_id = p.id
                    min_gini_val = p.gini_val
                if p.left:
                    node_que.put(p.left)
                if p.right:
                    node_que.put(p.right)
            print('cut branch, min_gini_val=' + str(min_gini_val) + ', node\'s id=' + str(min_gini_id))
            new_tree_root = self._gen_new_tree(tree_root, min_gini_id)
            gen_tree_roots.append(new_tree_root)
            print('current candidate tree\'s count=' + str(len(gen_tree_roots)))
            self._clean_tree(tree_root)  # 清空占用空间
            tree_root = copy.deepcopy(new_tree_root)
        # 在验证集上选择准确率最高的决策树
        print('finding the best cart tree.....')
        max_precision = 0.
        for cart_tree_root in gen_tree_roots:
            precision = self._validate_carttree(cart_tree_root, type_count, validate_file_type, validate_dir)
            if precision > max_precision:
                max_precision = precision
                print('current get best precision='+str(max_precision))
                self.cart_root = cart_tree_root  # 更新最优决策树
        print('the best cart tree found.....')
        print('tree structure is:')
        traverse(self.cart_root,0)


def traverse(cart_node: TreeNode, depth: int):
    if not cart_node:
        return
    print('   ' * depth + str(cart_node))
    if cart_node.left:
        traverse(cart_node.left, depth + 1)
    if cart_node.right:
        traverse(cart_node.right, depth + 1)


if __name__ == '__main__':
    if not path.exists('cart.model'):
        cart = CART('classify_train', 'stopwords.txt')
        traverse(cart.cart_root, 0)
        cart.cut_branch('classify_validate')
        with open('cart.model', 'wb') as fw:
            pickle.dump(cart, fw)
    else:
        with open('cart.model', 'rb') as fr:
            cart = pickle.load(fr)
    cart.file_classify('classify_test')
