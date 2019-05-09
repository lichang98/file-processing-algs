# -*- coding:utf-8 -*-
"""
    使用人民日报语料库测试分词算法的召回率与准确率
"""
from os import path
from queue import Queue
from sys import stdout

CIKU = path.join(path.abspath('.\\res'), 'cityu_training_words.utf8')
DOC = path.join(path.abspath('.\\res'), 'cityu_test.utf8')
RESULT = path.join(path.abspath('.\\res'), 'result.utf8')
RESULT_TRIE = path.join(path.abspath('.\\res'), 'trie_result4.utf8')
RESULT_BICUTREV = path.join(path.abspath('.\\res'), 'bicut_rev_result4.utf8')
LOG = path.join(path.abspath('.'), 'run.log')


class WordBiCut:
    """
    基于词典分词的整词二分算法
    1. 使用汉字国标区位码创建词语首字哈希表
    2. 哈希表项包含以该字开头的词语个数，指针
    3. 指针指向的有序列表可使用二分查找
    例：{'字':{'pre_cnt':1,'words':['字词']}, ...}
    """
    hash_table = {}
    MAX_WORD_LEN = 17  # 预计最大词语长度

    def __init__(self):
        # 建立整词二分哈希表
        with open(CIKU, encoding='utf-8', mode='r') as f:
            for word in f.readlines():
                word = word.strip()
                prefix = word[:1]  # 提取词语首字
                if prefix not in self.hash_table.keys():
                    self.hash_table.update({prefix: {'pre_cnt': 0, 'words': []}})
                self.hash_table[prefix]['pre_cnt'] += 1
                self.hash_table[prefix]['words'].append(word)
        # 对每个表项的词语列表进行排序
        for val in self.hash_table.values():
            val['words'].sort()

    def __bifind(self, lst, item):
        """
        在lst 中使用二分查找的方法查找item
        :param lst:
        :param item:
        :return: bool
        """
        begin = 0
        end = len(lst)
        while begin < end - 1:
            mid = int((begin + end) / 2)
            if lst[mid] == item:
                return True
            elif item < lst[mid]:
                end = mid
            else:
                begin = mid
        return True if lst[begin] == item else False

    def word_cut(self, srcfile, resfile):
        """
        对文件进行整词二分方法进行分词操作
        :param srcfile:
        :param resfile:
        :return:
        """
        with open(srcfile, encoding='utf-8', mode='r') as f, open(resfile, encoding='utf-8', mode='w') as fres \
                , open(LOG, encoding='utf-8', mode='w') as log:
            for line in f.readlines():
                result = []
                line = line.strip()
                begin = 0
                end = begin + self.MAX_WORD_LEN
                while begin < len(line):
                    if line[begin:begin + 1] not in self.hash_table.keys():
                        result.append(line[begin:begin + 1])
                        print('DEBUG:prefix={},word is {}'.format(line[begin:begin + 1], line[begin:end]), file=log)
                        begin += 1
                        end = begin + self.MAX_WORD_LEN if begin + self.MAX_WORD_LEN < len(line) else len(line)
                    else:
                        while end - begin > 1:
                            # if line[begin:end] in self.hash_table[line[begin:begin + 1]]['words']:
                            if self.__bifind(self.hash_table[line[begin:begin + 1]]['words'], line[begin:end]):
                                result.append(line[begin:end])
                                print('DEBUG:prefix={},word is {}'.format(line[begin:begin + 1], line[begin:end]),
                                      file=log)
                                begin = end
                                end = begin + self.MAX_WORD_LEN if begin + self.MAX_WORD_LEN < len(line) else len(line)
                                break
                            else:
                                end -= 1
                        if end - begin == 1:
                            result.append(line[begin:end])
                            print('DEBUG:prefix={},word is {}'.format(line[begin:begin + 1], line[begin:end]), file=log)
                            begin = end
                            end = begin + self.MAX_WORD_LEN if begin + self.MAX_WORD_LEN < len(line) else len(line)
                # 将读取的一行分词的结果写入文件中
                fres.write("  ".join(result) + '\n')


class WordBiCutRev:
    """
    基于整词二分的逆向最大匹配算法
    结构同正向的整词二分算法
    1. 使用汉字国标区位码创建词语首字哈希表
    2. 哈希表项包含以该字开头的词语个数，指针
    3. 指针指向的有序列表可使用二分查找
    例：{'字':{'words':['字词']}, ...}
    """
    hash_table = {}
    MAX_WORDLEN = 17  # 预计最大词语长度

    def __init__(self):
        with open(CIKU, encoding='UTF-8', mode='r') as fr:
            for word in fr.readlines():
                word = word.strip()
                if word[:1] not in self.hash_table.keys():
                    self.hash_table.update({str(word[:1]): {'words': []}})
                self.hash_table[word[:1]]['words'].append(word)
        # 对列表中的词语进行排序
        for word_lst in self.hash_table.values():
            word_lst['words'].sort()

    def cutword(self, srcfile, destfile):
        """
        对srcfile 文件中的内容进行分词，并将结果保存在destfile中
        :param srcfile:
        :param destfile:
        :return:
        """
        with open(srcfile, encoding='utf-8', mode='r') as fr, open(destfile, encoding='utf-8', mode='w') as fw:
            for line in fr.readlines():
                line = line.strip()
                result = []
                end = len(line)
                begin = end - self.MAX_WORDLEN if end - self.MAX_WORDLEN >= 0 else 0
                while end > 0:
                    while line[begin:begin + 1] not in self.hash_table.keys() and begin < end:
                        begin += 1
                    # print('DEBUG: begin={}, end={},str{}'.format(begin, end, line[begin:end]))
                    if begin == end:
                        result.append(line[end - 1:end])
                        end -= 1
                        begin = end - self.MAX_WORDLEN if end - self.MAX_WORDLEN >= 0 else 0
                    else:
                        if line[begin:end] in self.hash_table[line[begin:begin + 1]]['words']:
                            result.append(line[begin:end])
                            end = begin
                            begin = end - self.MAX_WORDLEN if end - self.MAX_WORDLEN >= 0 else 0
                        else:
                            begin += 1
                result.reverse()
                print("  ".join(result), file=fw)


class TreeNode:
    """
    TRIE 索引树结构的节点
    节点为表结构{'litter':nexttreenode,'litter2':nextnode2,...}
    """

    def __init__(self, word=''):
        """
        对词语word 建立索引树结构
        :param word:
        """
        self.node_table = {}
        index = 0
        ndtbl = self.node_table
        while index < len(word):
            treenode = TreeNode()
            ndtbl.update({word[index:index + 1]: treenode})
            ndtbl = treenode.node_table
            index += 1

    def update(self, word):
        """
        对索引树结构进行更新
        :param word:
        :return:
        """
        index = 0
        ndtbl = self.node_table
        while index < len(word):
            if word[index:index + 1] in ndtbl.keys():
                ndtbl = ndtbl[word[index:index + 1]].node_table
                index += 1
            else:
                ndtbl.update({word[index:index + 1]: TreeNode(word[index + 1:])})

    def traverse(self, logfile, depth=0, ):
        """
        对索引树结构进行遍历
        :param depth:
        :return:
        """
        for key, val in self.node_table.items():
            print('  ' * depth + str(key), file=logfile)
            val.traverse(logfile, depth=depth + 1)


class TrieTreeCut:
    """
    TRIE 索引树中文分词算法
    """
    hash_table = {}

    def __init__(self):
        with open(CIKU, encoding='UTF-8', mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if line[:1] not in self.hash_table.keys():
                    self.hash_table.update({str(line[:1]): TreeNode(line[1:])})
                else:
                    treenode = self.hash_table[line[:1]]
                    treenode.update(line[1:])

    def traverse(self):
        """
        遍历索引结构
        :return:
        """
        with open(LOG, encoding='UTF-8', mode='w') as f:
            for key, val in self.hash_table.items():
                print(str(key), file=f)
                val.traverse(f, depth=1)

    def cutword(self, srcfile, destfile):
        """
        对srcfile 进行分词操作，并将分词结果保存在destfile中
        :param srcfile:
        :param destfile:
        :return:
        """
        with open(srcfile, encoding='UTF-8', mode='r') as fr, open(destfile, encoding='UTF-8', mode='w') as fw:
            for line in fr.readlines():
                line = line.strip()
                begin = 0
                end = 1
                result = []
                while begin < len(line):
                    while line[begin:end] not in self.hash_table.keys() and begin < len(line):  # assert begin+1==end
                        result.append(line[begin:end])
                        begin += 1
                        end += 1
                    if begin >= len(line):
                        break
                    treenode = self.hash_table[line[begin:end]]
                    ndtbl = treenode.node_table
                    end += 1
                    while ndtbl and end <= len(line):
                        if line[end - 1:end] in ndtbl.keys():
                            ndtbl = ndtbl[line[end - 1:end]].node_table
                            end += 1
                        else:
                            end -= 1
                            result.append(line[begin:end])
                            begin = end
                            end += 1
                            break
                    if begin < len(line) and end >= len(line):
                        result.append(line[begin:])
                        begin = len(line)
                    elif not ndtbl:
                        end -= 1
                        result.append(line[begin:end])
                        begin = end
                        end += 1
                print("  ".join(result), file=fw)  # 将结果输出到文件中


if __name__ == '__main__':
    pass
#    log = path.join(path.abspath('.'), 'run.log')
#    wcut = WordBiCut()
#    wcut.word_cut(DOC, RESULT)
#    with open(log, mode='w', encoding='utf-8') as f:
#        for item in wcut.hash_table.items():
#            print(item, file=f)

    # treenode = TreeNode('abslopf')
    # nodetbl=treenode.node_table
    # while nodetbl:
    #     print(nodetbl)
    #     nodetbl = nodetbl[list(nodetbl.keys())[0]].node_table

#    trie = TrieTreeCut()
#    trie.cutword(DOC, RESULT_TRIE)
    # trie.traverse()

    wcut_rev = WordBiCutRev()
    wcut_rev.cutword(DOC, RESULT_BICUTREV)
