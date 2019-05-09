# -*- coding:utf-8 -*-
"""
n最短路径分词算法
基于dijkstra最短路径算法
结构描述：
        1. 节点处记录最短路径，边表示字符串上的汉字
        2. 每个节点记录n个最短路径上的前驱节点以及从起始位置的长度
边的权值使用TF-IDF:
    TF : 词语在文章中出现的次数/文章的总词数
    IDF: log(语料库文档总数/(包含该词的文档数+1))

"""
from os import path
import copy
import re

CIKU = path.join(path.abspath('.\\res'), 'cityu_training_words.utf8')
DOC = path.join(path.abspath('.\\res'), 'cityu_test.utf8')
RESULT_NPATH = path.join(path.abspath('.\\res'), 'npath_result4.utf8')
LOG = path.join(path.abspath('.'), 'run.log')
IDF_FILE = path.join(path.abspath('.\\res'), 'words_idf.txt')


class Graph:
    """
    图结构{vertext:[linkvertexs], ...}
    """

    def __init__(self, connections):
        """
        初始化
        :param connections: arcs: [(,),(,), ...]
        """
        self.vertexs = {}
        for s, d in connections:
            if s not in self.vertexs.keys():
                self.vertexs.update({s: []})
            if d not in self.vertexs.keys():
                self.vertexs.update({d: []})
            if d not in self.vertexs[s]:
                self.vertexs[s].append(d)

    def dijkstra(self, v0, ve):
        """
        dijkstra算法
        :param v0 起始点
        :param ve 终点
        :return: path[],dis
        """
        dis = [99999 for i in range(len(self.vertexs))]
        dis[v0] = 0
        rg = len(self.vertexs)
        visited = [False for i in range(rg)]
        spath = []
        pre_nodes = {}  # {nodei: it's prenode, ...}
        for i in range(rg):
            min_dis = 99999
            min_dis_vert = -1
            for i in range(rg):
                if dis[i] < min_dis and not visited[i]:
                    min_dis = dis[i]
                    min_dis_vert = i
            u = min_dis_vert
            visited[u] = True
            if u in self.vertexs.keys():
                for v in self.vertexs[u]:
                    if v in self.vertexs[u] and dis[u] + 1 < dis[v]:
                        dis[v] = dis[u] + 1
                        pre_nodes[v] = u
        haspath = True
        node = ve
        while node != v0:
            if node not in pre_nodes.keys():
                haspath = False
                break
            else:
                node = pre_nodes[node]
        if haspath:
            node = ve
            while node != v0:
                spath.append(node)
                node = pre_nodes[node]
            spath.append(v0)
            spath.reverse()
            dis_res = dis[ve]
        else:
            spath = []
            dis_res = 99999
        # if ve in pre_nodes.keys():
        #     node = ve
        #     while node != v0:
        #         spath.append(node)
        #         node = pre_nodes[node]
        #     spath.append(v0)
        #     spath.reverse()
        return spath, dis_res


class ShortestPath:
    """
    求取最短路径
    """
    words = []

    def __init__(self, string, dictionary):
        """

        :param string:
        :param dictionary: 词典
        """
        self.string = string
        self.connections = []  # [(a,d), (b,c), ()...]
        self.words = dictionary
        self.__initGraph()
        self.graph = Graph(self.connections)

    # def __readWords(self, file):
    #     if not self.words:
    #         with open(file, encoding='utf-8', mode='r') as fr:
    #             for line in fr.readlines():
    #                 self.words.append(line.strip())

    def __initGraph(self):
        """
        初始化词语切分图
        :return:
        """
        for i in range(len(self.string)):
            self.connections.append((i, i + 1))
        for i in range(len(self.string)):
            for j in range(i + 2, len(self.string) + 1):
                if self.string[i:j] in self.words:
                    self.connections.append((i, j))

    def ksp(self, k, ve):
        """
        计算k个最短路径
        起始节点为0号节点
        :param ve 终止节点
        :return: a_paths 最短路径集合
        """
        a_paths = []  # [(len1,[v0,v1,...]), (), ....]
        b_paths = []
        spath, dis = self.graph.dijkstra(0, ve)
        a_paths.append((dis, spath))
        if len(spath) > 2:
            for i in range(k):
                del_connection = copy.copy(self.connections)
                rg = len(spath)
                for i in range(rg - 1):
                    del_connection = copy.copy(self.connections)
                    if (spath[i], spath[i + 1]) in self.connections:
                        del_connection.remove((spath[i], spath[i + 1]))
                        dgraph = Graph(del_connection)
                        tmp_spath, tmp_dis = dgraph.dijkstra(spath[i], ve)
                        # print('DEBUG:')
                        # print('ve={}, \ngraph={}\ndis={}, spath={}'.format(ve, dgraph.vertexs, dis, spath))
                        # print('*******************************************')
                        if (tmp_dis, tmp_spath) not in b_paths and tmp_dis != 99999:
                            tmp_spath = spath[:i] + tmp_spath
                            tmp_dis = len(tmp_spath) - 1
                            b_paths.append((tmp_dis, tmp_spath))
                # 从b_paths中选择len最小的，删除并加入到a_paths中，进行下一次迭代
                if not b_paths:  # 如果b_path为空
                    break
                else:
                    min_dis = b_paths[0][0]
                    min_spath = b_paths[0][1]
                    for k, v in b_paths:
                        if k < min_dis:
                            min_dis = k
                            min_spath = v
                    b_paths.remove((min_dis, min_spath))
                    if (min_dis, min_spath) not in a_paths:
                        a_paths.append((min_dis, min_spath))
                    spath = min_spath
        return a_paths


def cut_words(srcfile, destfile, n=1):
    """
    分词
    :param srcfile:
    :param destfile:
    :param n: n_path 参数
    :return:
    """
    words = []
    with open(CIKU, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            words.append(line.strip())
    with open(srcfile, encoding='UTF-8', mode='r') as fr, open(destfile, encoding='utf-8', mode='w') as fw:
        for line in fr.readlines():
            line = line.strip()
            if not line:
                print('',file=fw)
                continue
            sp = ShortestPath(line, words)
            spaths = sp.ksp(n, len(line))
            cut_indexs = spaths[0][1]
            result = []
            for i in range(len(cut_indexs) - 1):
                result.append(line[cut_indexs[i]:cut_indexs[i + 1]])
#            print('  '.join(result))
            print('  '.join(result), file=fw)


if __name__ == '__main__':
    cut_words(DOC, RESULT_NPATH)
    # string = "他说的确实在理"
    # sp = ShortestPath(string)
