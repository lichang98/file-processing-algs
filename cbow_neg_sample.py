# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:29:50 2019
词向量表示
使用基于负采样的cbow 模型
@author: 李畅
"""
import sys
import numpy as np
import time
import math
from numpy.linalg import multi_dot
import argparse
import io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

class CBow:
    M_split=1e8 # 将词汇表等分的份数
    def __init__(self,inputfile,vec_dim=5,window_size=2,step=2,neg=10,max_iter=3):
        """
        初始化
        @param inputfile: 分词后的文件
        @param vec-dim : 词向量的维度大小
        @param window_size: cbow 模型上下文的大小
        @param step: 步长的大小
        @param neg :负采样的个数
        @param max_iter: 迭代次数
        """
        self.vocab=self._file_content_to_vocab(inputfile)
#        self.word_freqs=self._freq_counter(self.vocab)
        self.words,self.freqs=self._freq_counter(self.vocab)
        self.window_size=int(window_size)
        self.vec_dim=int(vec_dim)
        self.step=step
        self.max_iter=max_iter
        # 对于词汇表中每个词的模型参数
        self.model_theta=np.random.rand(len(self.words)*self.vec_dim)+1e-6
        self.model_theta=np.mat(self.model_theta)
        self.model_theta.resize((len(self.words),self.vec_dim))
        for i in range(np.shape(self.model_theta)[0]):
            for j in range(np.shape(self.model_theta)[1]):
                self.model_theta[i,j]=self._softmax(self.model_theta[i,j])
        # 每个词语的词向量，大小为 词汇表大小*制定的词向量大小
        self.word_vecs=np.mat(np.random.ranf(self.vec_dim*len(self.words))+1e-6)
        self.word_vecs.resize((len(self.words),self.vec_dim))
        for i in range(np.shape(self.word_vecs)[0]):
            for j in range(np.shape(self.word_vecs)[1]):
                self.word_vecs[i,j]=self._softmax(self.word_vecs[i,j])
        self.neg=neg
    
    def _file_content_to_vocab(self, filename):
        """
        将文件中的词语转换为numpy 词汇表矩阵
        @param filename: 分词后的文件
        @return numpy matrix 1*n矩阵
        """
        vocab_lst=[]
        with open(filename,encoding='utf-8',mode='r') as fr:
            for line in fr.readlines():
                line=line.strip()
                if line:
                    if line.find(' ') ==-1:
                        vocab_lst.append(line)
                    else:
                        vocab_lst.extend([item.strip() \
                                          for item in line.split() \
                                          if item.strip()])
        return np.mat(vocab_lst)
    
    def _freq_counter(self, datamatrix):
        """
        计算每一个词的频率
        @param datamatrix: numpy matrix 类型的 1*n 的词语表
        @return : m*2的numpy 矩阵，每一行为[词语，频率] 频率范围为0.0~1.0
        """
        word_freq={}
        for word in self.vocab.A[0]:
            if word not in word_freq.keys():
                word_freq.update({word:1.0})
            else:
                word_freq[word]+=1.0
        total_cnt=np.shape(self.vocab)[1]
        words=list(word_freq.keys())
        freqs=np.mat(np.zeros(len(words)))
        index=0
        for freq in word_freq.values():
            freqs[0,index]=freq*1.0/total_cnt
            index+=1
        return words,freqs
    
    def model_train(self):
        """
        cbow 模型训练
        """
        for itr in range(self.max_iter):
            for i in range(len(self.words)):
                self._ascend_iter(i)
            print('iter={} .. '.format(itr),end='')
    
    def _ascend_iter(self,w0_index):
        """
        梯度上升迭代
        @param w0_index: 当前正样本index
        @return: 返回当前迭代的梯度值
        """
        context,w0_index,negs=self._get_train_datas(w0_index)
        e=.0
        sum_x_context=np.zeros(self.vec_dim) # 计算上下文的词语向量之和
        for index in list(context[0].A[0]):
            sum_x_context += self.word_vecs[index].A[0]
        self.word_vecs[w0_index]=sum_x_context*1.0/(2*self.window_size)
        for i in list(negs.A[0]):
            q=np.sum(self.word_vecs[w0_index].A[0]*self.model_theta[i].A[0])
            q=self._softmax(q)
            g=self.step*(1-q)
            e=e+g*self.model_theta[i].A[0]
            self.model_theta[i]=self.model_theta[i].A[0]+g*self.word_vecs[
                    w0_index].A[0]
#            print('w0_ind={}, neg={}, q={}, e={}'.format(w0_index,i,q,e))
        # 更新上下文的词向量
        for j in list(context.A[0]):
            self.word_vecs[j]=self.word_vecs[j].A[0]+e
        return e
    
    def _get_train_datas(self,w0_index):
        """
        在对每个词语负采样后，迭代操作前，获取待处理的样本数据
        @param w0_index: 正样例的下标
        @return: context: 当前正样本w0的上下文
                w0: 当前正样本
                negs: 采样得到的负样本
        """
        negs=self._neg_sample(w0_index) # 获得负采样
        context=[]
        for i in range(w0_index-self.window_size,w0_index+self.window_size+1):
            if i!= w0_index and i >=0 and i < len(self.words):
                context.append(i)
        context=np.mat(context)
        return context,w0_index,negs
    
    def _neg_sample(self,w0_index):
        """
        随机负采样neg 个词语
        @param w0_index: 当前采样的中心词
        @return: 返回采样的词语在词汇表中的index, numpy mat 1*neg
        """
        neg_lst=[]
        for _ in range(self.neg):
            neg_lst.append(self._neg_sample_one(w0_index))
        return np.mat(neg_lst)
    
    def _neg_sample_one(self,w0_index):
        """
        随机的采样一个词
        @param w0_index: 当前负采样的中心词, 采样的负样本不为w0
        @return : 返回负采样的词在词汇表中的index
        """
        rand_m=np.random.randint(0,self.M_split+1,size=1,dtype=np.int32)[0]
        rand_m = rand_m *1.0/self.M_split  # 采样频率值
        # 找到该频率值对应的词语
        freq_sum=0.0
        ans_index=0
        for i in range(len(self.words)):
            if rand_m >= freq_sum and rand_m < freq_sum + self.freqs[0,i] \
                                                and i != w0_index:
                ans_index=i
                break
            else:
                freq_sum += self.freqs[0,i]
        return ans_index 
    
    def _softmax(self,inx):
        """
        softmax 归一化函数
        """
        return 1.0/(1+math.exp(0-inx))

if __name__=='__main__':
#    filename=sys.argv[1]
#    filename='./pku_word_embed_test.txt'
    parser=argparse.ArgumentParser()
    parser.add_argument('-f',required=True,\
                        dest='file',
                        help='the target file to build vec')
    parser.add_argument('-d',required=True,
                        dest='dim',
                        help='the dimension of vector')
    parser.add_argument('-z',required=True,
                        dest='win_size',
                        help='the window size of context')
    parser.add_argument('-s',required=True,
                        dest='step',
                        help='learn step size')
    parser.add_argument('-n',required=True,
                        dest='neg',
                        help='the number of neg samples')
    parser.add_argument('-r',required=True,
                        dest='max_iter',
                        help='the epoch of the iteration')
    args=parser.parse_args()
    cbow=CBow(args.file,vec_dim=int(args.dim),window_size=int(args.win_size),
              step=int(args.step),neg=int(args.neg),max_iter=int(args.max_iter))
    cbow.model_train()
    for i in range(len(cbow.words)):
        print('word is:{}\nvector={}'.format(cbow.words[i],cbow.word_vecs[i,:]))
#    # 绘制词向量图
#    fg=plt.figure()
#    ax=plt.subplot(111)
#    for i in range(len(cbow.words)):
#        ax.scatter(cbow.word_vecs[i,0],cbow.word_vecs[i,1],c='b')
#        ax.annotate(cbow.words[i],(cbow.word_vecs[i,0],cbow.word_vecs[i,1]))
#    plt.show()












