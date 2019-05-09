# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:01:30 2019
使用词语相似度计算的方法处理未登录词
区别于使用平滑系数
@author: 李畅
"""

from os import path,listdir,makedirs,rename,chmod
import pickle
from math import pow,log,exp
import argparse
import time
import sys
from gensim.models import word2vec

class Naive_bayes:
    def __init__(self,file_type_num,train_dir):
        """
        @param file_type_num: 文本的类别个数
        @param train_dir: 用于分类训练的目录
        """
        self.type_prob=[]
        self.type_names=[]
        self.type_feat_cnt={} # {文档类i:{特征j: 出现次数, ...}, ...}
        self.all_feats=set() # 特征的所有的可能的取值
        self.doc_feat_prob_cnt={} # {文档i:{特征j:(条件概率值，出现次数)}}
        assert path.exists('tiny_vol_wordvec.model'),'word2vec 模型文件不存在'
        self.word_sim_model=word2vec.Word2Vec.load('tiny_vol_wordvec.model')
        self.train_dir=train_dir
        self._cal_type_prob()
        self._cal_type_feats()
    
    def _cal_type_prob(self):
        """
        统计各类文档出现的概率
        """
        print('all type docs calculating.....')
        for subdir in listdir(self.train_dir):
            self.type_names.append(subdir)
            self.type_prob.append(len(listdir(path.join(self.train_dir,\
                                            subdir))))
        all_file_cnt=sum(self.type_prob)
        self.type_prob=[item * 1.0/all_file_cnt for item in self.type_prob]
        print('calculating finish.........',flush=True)
    
    def _cal_type_feats(self):
        """
        计算每个类别的文档，每个特征的出现次数
        """
        progres=0
        for subdir in listdir(self.train_dir):
            self.type_feat_cnt.update({subdir:{}})
            for doc in listdir(path.join(self.train_dir,subdir)):
                progres +=1
                if progres % 100==0:
                    print('train file , solved:{}'.format(progres),flush=True)
                with open(path.join(self.train_dir,subdir,doc),encoding=\
                          'utf-8',mode='r') as fr:
                    for line in fr.readlines():
                        line=line.strip()
                        if line:
                            line=[item.strip() for item in line.split() if item.strip()]
                            for feat in line:
                                self.all_feats=self.all_feats | {feat}
                                if feat not in self.type_feat_cnt[subdir].keys():
                                    self.type_feat_cnt[subdir].update({feat:1.0})
                                else:
                                    self.type_feat_cnt[subdir][feat]+=1.0
    
    
    def _prob_soft_migrate(self,prob):
        """
        对概率值进行平滑处理并迁移至正数区间
        @param prob: 概率值
        """
        prob=1.0/(1+exp(0-prob))
        prob += 3.0
        return prob
    
    def _argmax_sim_feat(self,type_name,word):
        """
        计算当前词word 与训练数据集中相似度最大的词语
        @param type_name: 当前与word比较的类别
        @param word : 当前比较相似度的词语
        @return: 返回该类文档中与当前词语相似度最大的词语
        """
        max_sim_val=float('-inf')
        max_sim_word='' # 相似度最大的词语
        for feat in self.type_feat_cnt[type_name].keys():
            if self.word_sim_model.similarity(feat,word) > max_sim_val:
                max_sim_val=self.word_sim_model.similarity(feat,word)
                max_sim_word = feat
        return max_sim_word
    
    def cal_condition_prob(self,testdir):
        """
        计算每个文档， 每个特征的条件概率值以及出现的次数
        @param testdir: 测试文档目录路径
        """
        for type_name in self.type_names:
            makedirs(path.join(testdir,type_name),mode=755)
            chmod(path.join(testdir,type_name),0o755)
        progres=0
        for doc in listdir(testdir):
            progres +=1
            if progres % 100==0:
                print('test file, solved:{}'.format(progres),flush=True)
            self.doc_feat_prob_cnt.update({doc[:6]:{}})
            with open(path.join(testdir,doc),encoding=\
                      'utf-8',mode='r') as fr:
                for line in fr.readlines():
                    line=line.strip()
                    if line:
                        line=[item.strip() for item in line.split() if item.strip()]
                        for feat in line:
                            if feat not in self.doc_feat_prob_cnt[doc[:6]].keys():
                                self.doc_feat_prob_cnt[doc[:6]].update({feat:[.0,.0]})
                            self.doc_feat_prob_cnt[doc[:6]][feat][1] +=1.0
            # 计算该文档在每个类别下的条件概率
            max_prob=-1.0
            max_prob_type=''
            lamda=log(len(self.all_feats)) # 平滑系数
            for type_name in self.type_names:
                for feat in self.doc_feat_prob_cnt[doc[:6]].keys():
                    # 如果训练集中没有出现该词语，则使用相似度最大的那个词进行替换
                    if feat not in self.type_feat_cnt[type_name].keys():
                        feat=self._argmax_sim_feat(type_name,feat)
                    self.doc_feat_prob_cnt[doc[:6]][feat][0]=log(\
                        (self.type_feat_cnt[type_name][feat]+lamda)*1.0/
                        (log(len(self.type_feat_cnt[type_name].keys())+\
                        log(len(self.all_feats)))))
                    self.doc_feat_prob_cnt[doc[:6]][feat][0] = \
                        self._prob_soft_migrate(self.doc_feat_prob_cnt[doc[:6]]\
                                                [feat][0])
#                    self.doc_feat_prob_cnt[doc[:6]][feat][0]=log(\
#                        ((self.type_feat_cnt[type_name][feat] if \
#                         feat in self.type_feat_cnt[type_name].keys()\
#                         else 0) +lamda)*1.0/ \
#                        (log(len(self.type_feat_cnt[type_name].keys()))+ \
#                        log(len(self.all_feats))))
#                    self.doc_feat_prob_cnt[doc[:6]][feat][0]= \
#                        self._prob_soft_migrate(self.doc_feat_prob_cnt[doc[:6]]\
#                                                [feat][0])
                prob=log(self.type_prob[self.type_names.index(type_name)])
                prob=self._prob_soft_migrate(prob)
                for feat in self.doc_feat_prob_cnt[doc[:6]].keys():
                    prob = log(prob*pow(self.doc_feat_prob_cnt[doc[:6]][feat][0],
                                    int(self.doc_feat_prob_cnt[doc[:6]][feat][1])))
                    prob=self._prob_soft_migrate(prob)
#                print('type is {}, probability ={}'.format(type_name,prob))
                if prob > max_prob:
                    max_prob=prob
                    max_prob_type=type_name
            # 将文件移动至对应类别的文件夹下
#            print('{} ---> {}'.format(doc,max_prob_type))
            rename(path.join(testdir,doc),path.join(testdir,max_prob_type,doc))
    


if __name__=='__main__':
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument('-n',required=True,dest='type_num',help='type count')
        parser.add_argument('--train_dir',required=True,dest='train_dir',
                            help='the directory of type sub directories')
        parser.add_argument('--test_dir',required=True,dest='test_dir',
                            help='the directory of testfiles')
        args=parser.parse_args()
        print('model start training....',flush=True)
        start=time.time()
        nv_bayes=Naive_bayes(int(args.type_num),args.train_dir)
        print('model training finish....',flush=True)
        nv_bayes.cal_condition_prob(args.test_dir)
        elapse=(time.time()-start)
        print('elapse = {}'.format(elapse),flush=True)
        print('file classify finish....',flush=True)
    except (BrokenPipeError, IOError):
        print('BrokenPipeError caught',file=sys.stderr)
    print('Done',file=sys.stderr)
    sys.stderr.close()