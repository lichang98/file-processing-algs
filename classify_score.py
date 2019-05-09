# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:41:41 2019
对文本分类后的结果进行评价
@author: 李畅
"""
from os import listdir,path
import argparse

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ans_dir',required=True,dest='ans_dir',
                        help='The directory which contains the answer of classes')
    parser.add_argument('--test_dir',required=True,dest='test_dir',
                        help='The directory which contains the result of \
                        file classification')
    args=parser.parse_args()
    ans_dir=args.ans_dir
    test_dir=args.test_dir
    types=listdir(ans_dir)
    ans_type_docs={} # {类别：[文件1，文件2，...], ...}
    ans_doc_type={} # {文件：类别, ...}
    test_type_docs={}
    test_doc_type={}
    for t_name in types:
        ans_type_docs.update({t_name:[]})
        for doc in listdir(path.join(ans_dir,t_name)):
            ans_type_docs[t_name].append(doc)
            ans_doc_type[doc]=t_name
        test_type_docs.update({t_name:[]})
        for doc in listdir(path.join(test_dir,t_name)):
            test_type_docs[t_name].append(doc)
            test_doc_type[doc]=t_name
    prec_rates=[]
    recall_rates=[]
    for tname in types:
        true_clasfied=0
        wrong_clasfied=0
        for doc in ans_type_docs[tname]:
            if test_doc_type[doc]==tname:
                true_clasfied+=1
            else:
                wrong_clasfied+=1
        prec_rates.append(true_clasfied*1.0/(true_clasfied+wrong_clasfied))
        recall_rates.append(true_clasfied*1.0/(true_clasfied+len(\
                        test_type_docs[tname])-true_clasfied+1e-8))
    print('precision={},recall rate={}'.format(sum(prec_rates)*1.0/len(prec_rates),
          sum(recall_rates)*1.0/len(recall_rates)))
    
    
    
    
