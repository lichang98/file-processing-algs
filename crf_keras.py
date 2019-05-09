# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:20:31 2019
基于Keras 的条件随机场分词算法
@author: 李畅
"""

from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,LSTM,BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras_contrib.layers import CRF
from keras.callbacks import TensorBoard
import numpy as np
from os import path,rename,remove
import pickle

mcrf_datas='./mcrf.data' # 以二进制保存的mcrf类数据

class MCRF:
    def __init__(self,train_file,validate_file,model_file=\
            './crf_model.h5',embedding_dim=128,birnn_units=128,feat_filt_freq=3,\
            train_max_len=64,test_max_len=1024,batch_size=64,epochs=10):
        """
        @param train_file : 训练文件
        @param validate_file : 验证文件
        @param model_file: 模型文件,h5类型文件
        @param embedding_dim: 隐层维度
        @param birnn_units: RNN 单元个数
        @param feat_filt_freq: 将出现频次低于该值的特征过滤掉
        @param train_max_len: 训练集中的最大长度
        @param test_max_len: 测试文件一行最长的长度
        @param batch_size: 训练过程中的批次大小
        @param epochs: 训练的轮次
        """
        self.train_file=train_file
        self.validate_file = validate_file
        self.model_file=model_file
        self.feat_filt_freq=feat_filt_freq
        self.embedding_dim=embedding_dim
        self.birnn_units=birnn_units
        self.test_max_len=test_max_len
        self.batch_size=batch_size
        self.epochs=epochs
        self.tags=['B','M','E','S'] # 字符所有可能的标记
        self.max_len=train_max_len # 最长的一段文字的长度
        self.vocab=[]
        (self.train_x,self.train_y),(self.test_x,self.test_y)=self._preprocess()
        pass
    
    def _process_data(self, data):
        """
        数据处理
        """
        word_index=dict((w,i) for i,w in enumerate(self.vocab))
        x=[[word_index.get(w[0], 1) for w in s] for s in data] # 对未登录词，index=0
        y_chunk=[[self.tags.index(w[1]) for w in s] for s in data]
        x= pad_sequences(x,self.max_len) # 在 x 的左侧填充0，使得各个sample的长度相同
        y_chunk=pad_sequences(y_chunk,self.max_len,value=-1) # 左侧填充-1
        y_chunk=np.expand_dims(y_chunk,2) # 扩充维度，每个标记为一个list[]
        return x,y_chunk
    
    def _preprocess(self):
        """
        数据预处理
        """
        feat_freqs={}
        data=[] # 二维矩阵，一行为一段文字
        with open(self.train_file,encoding='utf-8',mode='r') as fr:
            sample_len=0 # 记录当前一段文字的长度
            one_sample=[]
            for line in fr.readlines():
                line=line.strip()
                if not line or len(line) < 3:
                    sample_len=0
                    data.append(one_sample)
                    one_sample=[]
                    continue
                sample_len +=1
                word,tag=line.split()
                one_sample.append([word,tag])
                if (word,tag) not in feat_freqs.keys():
                    feat_freqs.update({(word,tag):0})
                feat_freqs[(word,tag)] +=1
        # 特征值过滤
        self.vocab=[key[0] for key,val in feat_freqs.items() if val > \
                    self.feat_filt_freq]
        train_datas = self._process_data(data)
        del data,feat_freqs
        data=[]
        #读取测试文件
        with open(self.validate_file,encoding='utf-8',mode='r') as fr:
            one_example=[]
            for line in fr.readlines():
                line=line.strip()
                if not line or len(line) < 3:
                    data.append(one_example)
                    one_example=[]
                    continue
                word,tag = line.split()
                one_example.append([word,tag])
        test_datas=self._process_data(data)
        return train_datas,test_datas
    
    def _create_model(self):
        """
        创建训练使用的模型
        """
        model=Sequential()
        model.add(Embedding(len(self.vocab),self.embedding_dim,mask_zero=True))
        model.add(Bidirectional(LSTM(self.birnn_units//2, return_sequences=True)))
        crf=CRF(len(self.tags),sparse_target=True)
        model.add(crf)
        model.summary()
#        model.compile('adam',loss=crf.loss_function,metrics=[crf.accuracy])
        rms_prop=optimizers.RMSprop(lr=0.01,decay=1e-4)
        model.compile(optimizer=rms_prop,loss=crf.loss_function,metrics=\
                      [crf.accuracy])
        return model
    
    def train(self,model_exists=False):
        """
        模型训练
        """
        model=self._create_model()
        if not model_exists:
            # 创建tensorboard 回调
            tb_callback=TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    batch_size=64,
                    write_graph=True,
                    write_grads=True,
                    write_images=True,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None)
            model.fit(self.train_x, self.train_y, batch_size=self.batch_size,epochs=self.epochs,\
                 validation_data=[self.test_x,self.test_y],callbacks=[tb_callback])
            model.save(self.model_file)
        else:
            model.load_weights(self.model_file)
        return model
    
    def _predict_str_preprocess(self,string):
        """
        对待处理的字符串预处理
        """
        word_index=dict((w,i) for i,w in enumerate(self.vocab))
        x=[word_index.get(w[0], 1) for w in string]
        length = len(x)
        x=pad_sequences([x],self.test_max_len)
        return x,length
    
    def predict(self,string,model):
        """
        对序列的标记进行预测
        @param string: 待处理的一串字符
        @param model: 模型
        """
        string,length=self._predict_str_preprocess(string)
        raw=model.predict(string)[0][-length:]
        result=[np.argmax(row) for row in raw]
        result_tags=[self.tags[i] for i in result]
        return result,result_tags
    
    def predict_file(self,testfile):
        """
        对测试文件4词位标记
        """
        if not path.exists(self.model_file):
            model = self.train(model_exists=False)
        else:
            model = self.train(model_exists=True)
        tmpfile=path.join(path.dirname(testfile),'tmp')
        with open(testfile,encoding='utf-8',mode='r') as fr, open(tmpfile,\
                 encoding='utf-8',mode='w') as fw:
            prog=0
            for line in fr.readlines():
                prog +=1
                if prog % 100==0:
                    print('processing line{}'.format(prog))
                line=line.strip()
                if not line:
                    print('',file=fw)
                    continue
                _, tags=self.predict(line,model)
                line=list(line)
                # 对该行文字分词
                ans=[]
                tmpstr=''
                for i in range(len(line)):
                    if tags[i]=='S':
                        tmpstr=''
                        ans.append(line[i])
                    elif tags[i] == 'B':
                        tmpstr=''
                        tmpstr=line[i]
                    elif tags[i] == 'M':
                        tmpstr += line[i]
                    else:
                        tmpstr += line[i]
                        ans.append(tmpstr)
                        tmpstr=''
                print('  '.join(ans),file=fw)
        remove(testfile)
        rename(tmpfile,testfile)
        pass
    
if not path.exists(mcrf_datas):
    mcrf=MCRF('./pku_training.utf8','./pku_validate.utf8')
else:
    with open(mcrf_datas,mode='rb') as fr:
        mcrf=pickle.load(fr)
test_file='./pku_test.utf8'
mcrf.predict_file(test_file)
