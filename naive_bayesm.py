# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:31:51 2019
使用朴素贝叶斯对文本进行分类
@author: 李畅
"""
from os import path,listdir,makedirs,rename,chmod
import pickle
from math import pow,log,exp
import argparse
import time

class Naive_bayes:
    def __init__(self,file_type_num,train_dir,tf_idf,feat_filt=11):
        """
        @param file_type_num: 文本的类别个数
        @param train_dir: 用于分类训练的目录
        @param tf_idf: 保存有每个文档tf_idf 信息的文件
        @param feat_filt: 至多使用文档tf-idf前n个特征项
        """
        self.type_prob=[]
        self.type_names=[]
        self.type_feat_cnt={} # {文档类i:{特征j: 出现次数, ...}, ...}
        self.all_feats=set() # 特征的所有的可能的取值
        self.doc_feat_prob_cnt={} # {文档i:{特征j:(条件概率值，出现次数)}}
        self.train_dir=train_dir
        self.feat_filt=feat_filt
        self.tf_idf={}
        with open(tf_idf,mode='rb') as fr:
            self.tf_idf=pickle.load(fr)  # {file:[(word,tf*idf), ...], ..}
        self._cal_type_prob()
        self._cal_type_feats()
    
    def _cal_type_prob(self):
        """
        统计各类文档出现的概率
        """
        for subdir in listdir(self.train_dir):
            self.type_names.append(subdir)
            self.type_prob.append(len(listdir(path.join(self.train_dir,\
                                            subdir))))
        all_file_cnt=sum(self.type_prob)
        self.type_prob=[item * 1.0/all_file_cnt for item in self.type_prob]
    
    def _cal_type_feats(self):
        """
        计算每个类别的文档，每个特征的出现次数
        """
        for subdir in listdir(self.train_dir):
            self.type_feat_cnt.update({subdir:{}})
            for doc in listdir(path.join(self.train_dir,subdir)):
                with open(path.join(self.train_dir,subdir,doc),encoding=\
                          'utf-8',mode='r') as fr:
                    # 使用tf-idf 过滤特征
                    cur_feats=[]
                    flag=min(len(self.tf_idf[doc]),self.feat_filt)
                    for word_val in self.tf_idf[doc]:
                        self.all_feats=self.all_feats|{word_val[0]}
                        cur_feats.append(word_val[0])
                        flag -=1
                        if flag==0:
                            break
                    for line in fr.readlines():
                        line=line.strip()
                        if not line:
                            continue
                        line=[item.strip() for item in line.split() if item.strip()]
                        for feat in line:
                            if feat in cur_feats:
                                if feat in self.type_feat_cnt[subdir].keys():
                                    self.type_feat_cnt[subdir][feat] +=1.0
                                else:
                                    self.type_feat_cnt[subdir].update({feat:1.0})
    
    
    def _prob_soft_migrate(self,prob):
        """
        对概率值进行平滑处理并迁移至正数区间
        @param prob: 概率值
        """
        prob=1.0/(1+exp(0-prob))
        prob += 3.0
        return prob
    
    def cal_condition_prob(self,testdir):
        """
        计算每个文档， 每个特征的条件概率值以及出现的次数
        @param testdir: 测试文档目录路径
        """
        for type_name in self.type_names:
            makedirs(path.join(testdir,type_name),mode=755)
            chmod(path.join(testdir,type_name),0o755)
        for doc in listdir(testdir):
            if path.isdir(path.join(testdir,doc)):
                continue
            self.doc_feat_prob_cnt.update({doc[:6]:{}})
            with open(path.join(testdir,doc),encoding=\
                      'utf-8',mode='r') as fr:
                # 使用tf-idf 排序后的值，选择前N个特征
                flag=min(len(self.tf_idf[doc]),self.feat_filt)
                for word_val in self.tf_idf[doc]:
                    self.doc_feat_prob_cnt[doc[:6]].update({word_val[0]:[.0,.0]})
                    flag-=1
                    if flag==0:
                        break
                for line in fr.readlines():
                    line=line.strip()
                    if line:
                        line=[item.strip() for item in line.split() if item.strip()]
                        for feat in line:
                            if feat in self.doc_feat_prob_cnt[doc[:6]].keys():
                                self.doc_feat_prob_cnt[doc[:6]][feat][1] +=1.0
            # 计算该文档在每个类别下的条件概率
            max_prob=-1.0
            max_prob_type=''
            lamda=log(len(self.all_feats)) # 平滑系数
            for type_name in self.type_names:
                for feat in self.doc_feat_prob_cnt[doc[:6]].keys():
                    self.doc_feat_prob_cnt[doc[:6]][feat][0]=log(\
                        ((self.type_feat_cnt[type_name][feat] if \
                         feat in self.type_feat_cnt[type_name].keys()\
                         else 0) +lamda)*1.0/ \
                        (log(len(self.type_feat_cnt[type_name].keys()))+ \
                        log(len(self.all_feats))))
                    self.doc_feat_prob_cnt[doc[:6]][feat][0]= \
                        self._prob_soft_migrate(self.doc_feat_prob_cnt[doc[:6]]\
                                                [feat][0])
                prob=log(self.type_prob[self.type_names.index(type_name)])
                prob=self._prob_soft_migrate(prob)
                for feat in self.doc_feat_prob_cnt[doc[:6]].keys():
                    prob = log(prob*pow(self.doc_feat_prob_cnt[doc[:6]][feat][0],
                            self._prob_soft_migrate(self.doc_feat_prob_cnt\
                                                    [doc[:6]][feat][1])))
                    prob=self._prob_soft_migrate(prob)
#                print('type is {}, probability ={}'.format(type_name,prob))
                if prob > max_prob:
                    max_prob=prob
                    max_prob_type=type_name
            # 将文件移动至对应类别的文件夹下
#            print('{} ---> {}'.format(doc,max_prob_type))
            rename(path.join(testdir,doc),path.join(testdir,max_prob_type,doc))
    


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',required=True,dest='type_num',help='type count')
    parser.add_argument('--train_dir',required=True,dest='train_dir',
                        help='the directory of type sub directories')
    parser.add_argument('--tfidf',required=True,dest='t_idf',
                        help='the file name of tf_idf infors')
    parser.add_argument('--test_dir',required=True,dest='test_dir',
                        help='the directory of testfiles')
    parser.add_argument('--feat_num',required=True,dest='feat_filt',
                        help='how many feats used in file classify')
    args=parser.parse_args()
    start=time.time()
    print('naive bayes improved start training....')
    nv_bayes=Naive_bayes(int(args.type_num),args.train_dir,args.t_idf,feat_filt=\
                         int(args.feat_filt))
    nv_bayes.cal_condition_prob(args.test_dir)
    print('file classification finished .....')
    elapse=(time.time()-start)
    print('elapse = {}'.format(elapse))