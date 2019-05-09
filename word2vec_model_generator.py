# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:23:11 2019
用于生成tiny_vol 内所有文本的word2vec 模型
@author: 李畅
"""
from os import listdir,path
import sys
from gensim.models import word2vec

work_dirs=sys.argv[1:]

content=''
progress=0
for work_dir in work_dirs:
    for file in listdir(work_dir):
        progress+=1
        if progress % 100==0:
            print('solved file count={}'.format(progress))
        fr=open(path.join(work_dir,file),'r',encoding='utf-8')
        f_content=fr.read()
        fr.close()
        content += f_content
model=word2vec.Word2Vec(content,size=256,min_count=10)
model.save('tiny_vol_wordvec.model')
