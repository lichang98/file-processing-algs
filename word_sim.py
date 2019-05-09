# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 07:30:52 2019
使用HIT 词林， 构建基于路径与深度的同义词相似度计算模型
@author: 李畅
"""
import sys
from math import fabs
import pickle
from os import path

cilin_model='./cilin_model'

class Sim:
    def __init__(self,cilin):
        """
        @param cilin : 词林编码文件
        """
        self.cilin=cilin
        self.word_tree=dict()
        self.word_tree.update({'/':{}}) # 添加虚拟根节点
        self.word_paths={} # 记录每一个词从虚拟根节点出发的路径（每个词可能有多个）
        self.word_set=set()
        self._load_data()
        self.a=0.9 # 调节参数
        self.arc_w=[8.,6.,4.,1.5,0.5] # 连接各个层级的边的权重
    
    def _load_data(self):
        """
        从词林编码文件中读取编码，并构建层级树
        """
        with open(self.cilin,encoding='utf-8',mode='r') as fr:
            for line in fr.readlines():
                datas=line.split() # 以空格分隔
                code=datas[0]
                if code[:1] not in self.word_tree['/']:
                    self.word_tree['/'].update({code[:1]:{}})
                if code[1:2] not in self.word_tree['/'][code[:1]]:
                    self.word_tree['/'][code[:1]].update({code[1:2]:{}})
                if code[2:4] not in self.word_tree['/'][code[:1]][code[1:2]]:
                    self.word_tree['/'][code[:1]][code[1:2]]\
                            .update({code[2:4]:{}})
                if code[4:5] not in self.word_tree['/'][code[:1]][code[1:2]]\
                        [code[2:4]]:
                    self.word_tree['/'][code[:1]][code[1:2]][code[2:4]].\
                            update({code[4:5]:{}})
                if code[5:7] not in self.word_tree['/'][code[:1]][code[1:2]]\
                        [code[2:4]][code[4:5]]:
                    self.word_tree['/'][code[:1]][code[1:2]][code[2:4]]\
                            [code[4:5]].update({code[5:7]:{}})
                self.word_tree['/'][code[:1]][code[1:2]][code[2:4]][code[4:5]]\
                        [code[5:7]].update({'tag':code[7:]})
                self.word_tree['/'][code[:1]][code[1:2]][code[2:4]][code[4:5]]\
                        [code[5:7]].update({'words':[]})
                for word in datas[1:]:
                    self.word_tree['/'][code[:1]][code[1:2]][code[2:4]]\
                        [code[4:5]][code[5:7]]['words'].append(word)
                    self.word_set=self.word_set|{word}
                    if word not in self.word_paths.keys():
                        self.word_paths.update({word:[]})
                    self.word_paths[word].append(['/',code[:1],code[1:2],\
                                                 code[2:4],code[4:5],code[5:7],\
                                                 code[7:]]) # 最后一项为标记(#=@)
    
    
    def cal_sim(self,word1,word2):
        """
        计算两个词语之间的相似度
        @return: 返回词语相似度值
        """
        if word1 not in self.word_set or word2 not in self.word_set:
            print('word{} or {} not included in the \
                  dictionary'.format(word1,word2))
            return 0.0
        else:
            max_sim=.0
            for path1 in self.word_paths[word1]:
                for path2 in self.word_paths[word2]:
                    lcp_level=0
                    lcp_dict=self.word_tree['/']
                    flag=True
                    lcp_level=0
                    for i in range(1,6):
                        if path1[i] == path2[i] and flag:
                            lcp_level=i
                        else:
                            flag=False
                        lcp_dict=lcp_dict[path1[i]]
                    if lcp_level == 5:
                        # 词语编码相同,标记为 = 或 #
                        if path1[6] == '=':
                            return 1.0
                        else:
                            return 0.5
                    else:
                        branch1=int(path1[lcp_level+1]) if \
                            path1[lcp_level+1].isdigit() else ord(path1[lcp_level+1])
                        branch2=int(path2[lcp_level+1]) if \
                            path2[lcp_level+1].isdigit() else ord(path2[lcp_level+1])
                        k=fabs(branch1-branch2)
                        n=len(lcp_dict.keys())
                        beta=k*1.0/n*self.arc_w[lcp_level]
                        path_w=0.0
                        for i in range(lcp_level,5):
                            path_w += self.arc_w[i]
                        path_w *=2
                        lcp_depth_w=0.0
                        for i in range(lcp_level):
                            lcp_depth_w += self.arc_w[i]
                        sim=(lcp_depth_w+self.a)*1.0/(lcp_depth_w+self.a+\
                            path_w+beta)
                        max_sim= max_sim if max_sim > sim else sim
            return max_sim
        

if __name__=='__main__':
    if path.exists(cilin_model):
        with open(cilin_model,mode='rb') as fr:
            sim=pickle.load(fr)
    else:
        sim=Sim('cilin_ex.txt')
        print('word dictionary build finish...')
        with open(cilin_model,mode='wb') as fw:
            pickle.dump(sim,fw)
    sim_val = sim.cal_sim(sys.argv[1],sys.argv[2])
    print(sim_val)

