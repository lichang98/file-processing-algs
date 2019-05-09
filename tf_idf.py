# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:30:19 2019
对每个文档计算TF-IDF值
计算结果的保存格式为{文档编号:[(w1,tf-idf1), ...], ...}
@author: 李畅
"""
from os import listdir,path
import pickle
from math import log
import io
import sys
import re
from functools import cmp_to_key

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

def cmp_func(word_val1,word_val2):
    if len(word_val1[0]) < len(word_val2[0]):
        return 1
    elif len(word_val1[0]) > len(word_val2[0]):
        return -1
    elif word_val1[1] > word_val2[1]:
        return -1
    elif word_val1[1] < word_val2[1]:
        return 1
    else:
        return 0

if __name__=='__main__':
    file_tf_idf={}
    dirs=sys.argv[1:]
    # 计算每个词语的IDF值
    word_file_cnts={} # {wordi:[file1, file2, ...], ...}
    total_doc_cnt=0
    for directory in dirs:
        for file in listdir(directory):
            if re.match('[0-9]+\.txt',file):
                total_doc_cnt+=1
                with open(path.join(directory,file),\
                          encoding='utf-8',mode='r') as fr:
                    all_words=set()
                    for line in fr.readlines():
                        line=line.split()
                        line=[item.strip() for item in line if item.strip()]
                        for word in line:
                            all_words = all_words | {word}
                    for word in all_words:
                        if word not in word_file_cnts.keys():
                            word_file_cnts.update({word:[]})
                        word_file_cnts[word].append(file)
    idfs={} # {word: idf_val, ...}
    for word,file_lst in word_file_cnts.items():
        idfs.update({word:.0})
        idfs[word]=log(total_doc_cnt*1.0/(len(file_lst)+1))
    # 对每个文档计算TF值
    for directory in dirs:
        for file in listdir(directory):
            if re.match('[0-9]+\.txt',file):
                with open(path.join(directory,file),\
                          encoding='utf-8',mode='r') as fr:
                    file_all_words={} # {word:count, ...}
                    for line in fr.readlines():
                        line=line.split()
                        line=[item.strip() for item in line if item.strip()]
                        for word in line:
                            if word not in file_all_words.keys():
                                file_all_words.update({word:1})
                            else:
                                file_all_words[word] +=1
                tf_vals={} # {word:tf_val,  ...}
                for word,count in file_all_words.items():
                    tf_vals.update({word:count*1.0/len(
                            file_all_words.keys())})
                file_tf_idf.update({file:[]})
                for word,tf_val in tf_vals.items():
                    file_tf_idf[file].append((word,tf_val*idfs[word]))
    # 将计算得到的文档tf-idfs 保存到文件中
    for filename,val_pair in file_tf_idf.items():
        val_pair.sort(key=cmp_to_key(cmp_func))
#        print("file is {}: ".format(str(filename)),end='')
#        for i in range(min(7,len(val_pair))):
#            print("{}  ".format(val_pair[i]))
    with open('file_tfidf',mode='wb') as fw:
        pickle.dump(file_tf_idf,fw)