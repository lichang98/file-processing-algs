# -*- coding:utf-8 -*-
from os import path, rename, remove
import re
from functools import reduce
import pickle
import fileinput
import pathlib
import math

#RENMIN_1998_CORPUS = path.join(path.abspath('./res'), '1998_corpus.txt')
#RENMIN_2014_CORPUS = path.join(path.abspath('./res'), '2014_corpus.txt')
#CIYU_4PLACE_LABEL_1998 = path.join(path.abspath(
#    './res'), 'word_4place_label.txt')  # 4词位标记保存文件
CIYU_4PLACE_LABEL_SIGHAN = path.join(
    path.abspath('./res'), 'cityu_training_labeled.utf8')
TESTFILE = path.join(path.abspath('./res'), 'cityu_test.utf8')
RESULT_FILE = path.join(path.abspath('./res'), 'hmm_result4.utf8')
#MODEL_FILE = path.join(path.abspath('./res'), 'crf_model.data')


def corpus2place_label(srcfile, destfile):
    """
    将语料库文件转换为4词位标记文件
    :param srcfile:
    :param destfile:
    :return:
    """
    with open(srcfile, encoding='utf-8', mode='r') as fr, open(destfile, encoding='utf-8', mode='a') as fw:
        for line in fr.readlines():
            line = line.strip()
            lst = line.split(' ')
            for i in range(len(lst)):
                lst[i] = re.sub(r'\/.+ *$', "", lst[i])
                lst[i] = lst[i].strip()
            lst = list(filter(lambda x: x.strip() != '', lst))
            for i in range(len(lst)):
                itemlen = len(lst[i])
                if itemlen == 1:
                    print(lst[i] + ' S', file=fw)
                else:
                    print(lst[i][0] + ' B', file=fw)
                    for j in range(1, itemlen - 1):
                        print(lst[i][j] + ' M', file=fw)
                    print(lst[i][itemlen - 1] + ' E', file=fw)


def corpusQuan2Ban(file):
    """
    将文件中的数字由全角转换为半角
    :param file:
    :return:
    """
    newfilename = '.tmp'
    table = {'０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
             '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'}
    with open(file, encoding='utf-8', mode='r') as fr, open(path.join(path.abspath('./res'), newfilename),
                                                            encoding='utf-8', mode='w')as fw:
        for line in fr.readlines():
            newline = ''
            for char in list(line):
                if char in table.keys():
                    char = table[char]
                newline += char
            print(newline, file=fw, end='')
        # for line in fr.readlines():
        #     newline = ''
        #     for char in list(line):
        #         inside_code = ord(char)
        #         if inside_code == 12288:
        #             inside_code = 32
        #         elif inside_code >= 65281 and inside_code <= 65374:
        #             inside_code -= 65248
        #         newline += chr(inside_code)
        #     print(newline, file=fw,end='')
    remove(path.join(path.abspath('./res'), file))
    rename(path.join(path.abspath('./res'), newfilename),
           path.join(path.abspath('./res'), file))
    # for line in fileinput.input(file,openhook='utf-8',inplace=True):
    #     #     newline = ''
    #     #     for char in line:
    #     #         inside_code = ord(char)
    #     #         if inside_code == 12288:
    #     #             inside_code = 32
    #     #         elif inside_code >= 65281 and inside_code <= 65374:
    #     #             inside_code -= 65248
    #     #         newline += chr(inside_code)
    #     #     print(newline)


class Feature:
    """
    用于训练的词语特征模型
    """
    label2index = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    index2label = {0: 'B', 'M': 1, 'E': 2, 'S': 3}

    def __init__(self):
        self.total_cnt = 0  # 该字在语料库中出现的总次数
        self.status_cnt = [0, 0, 0, 0]  # 列表中分别对应当前字在语料库中以状态B M E S出现的次数
        self.status_possible = [0.0, 0.0, .0, .0]  # B M E S 的概率值
        self.transfer = [[.0, .0, .0, .0], [.0, .0, .0, .0], [.0, .0, .0, .0],
                         [.0, .0, .0, .0]]  # 4*4 矩阵，该字当前状态（B M E S ） 到下一个字的状态（B M E S ）的概率
        # {'B':{W_前1:概率，W_前2：概率}, 'M':{}, ...}
        self.pre_context = {'B': {}, 'M': {}, 'E': {}, 'S': {}}
        self.post_context = {'B': {}, 'M': {}, 'E': {}, 'S': {}}  # 后文关系


class Model:
    """
    训练的模型
    """

    def __init__(self):
        self.model = {}  # {字:feature, ....}

    def train_model(self, filename):
        """
        通过文件file中的词位标记信息进行模型的训练
        :param filename:
        :return:
        """
        progress = 0
        with open(filename, encoding='utf-8', mode='r') as fr:
            pre_word = ''
            pre_label = ''
            for line in fr.readlines():
                if line.strip() == '':
                    continue
                progress += 1
                if progress % 10000 == 0:
                    print('STATUS: 当前模型处理行数{}'.format(progress))
                try:
                    word, label = line.split(' ')
                except ValueError:
                    print('WRONG AT LINE {}'.format(progress))
                word = word.strip()
                label = label.strip()
                if word not in self.model.keys():
                    self.model.update({word: Feature()})
                self.model[word].total_cnt += 1
                self.model[word].status_cnt[Feature.label2index[label]] += 1
                # 转移矩阵
                # S/E -> B -> M/E , M/B->M -> M/E, M/B ->E -> B/S , E/S->S ->B/S
                if pre_word:
                    self.model[pre_word].transfer[Feature.label2index[pre_label]
                                                  ][Feature.label2index[label]] += 1
                    if word not in self.model[pre_word].post_context[pre_label].keys():
                        self.model[pre_word].post_context[pre_label].update({
                                                                            word: .0})
                    self.model[pre_word].post_context[pre_label][word] += 1
                    if pre_word not in self.model[word].pre_context[label].keys():
                        self.model[word].pre_context[label].update(
                            {pre_word: .0})
                    self.model[word].pre_context[label][pre_word] += 1
                pre_word = word
                pre_label = label
        print('词位标记文件读取结束....')
        # 计算概率值
        for word in self.model.keys():
            for i in range(4):
                self.model[word].status_possible[i] = float(self.model[word].status_cnt[i]) / float(self.model[word]
                                                                                                    .total_cnt) if self.model[word].total_cnt != 0 else 0
            # 计算转移矩阵的概率值
            for i in range(4):
                line_sum = float(
                    reduce(lambda x, y: x + y, self.model[word].transfer[i]))
                for j in range(4):
                    self.model[word].transfer[i][j] = float(self.model[word].transfer[i][j]) / float(
                        line_sum) if math.fabs(line_sum) > 1e-6 else 0
            # 计算上下文的概率值
            for key, val in self.model[word].pre_context.items():
                freq_sum = 0
                for freq in val.values():
                    freq_sum += freq
                for key_w in val.keys():
                    self.model[word].pre_context[key][key_w] = float(self.model[word].pre_context[key][key_w]) \
                        / float(freq_sum) if freq_sum != 0 else 0
            for key, val in self.model[word].post_context.items():
                freq_sum = 0
                for freq in val.values():
                    freq_sum += freq
                for key_w in val.keys():
                    self.model[word].post_context[key][key_w] = float(self.model[word].post_context[key][key_w]) \
                        / float(freq_sum) if freq_sum != 0 else 0


class CRF:
    """
    CRF 分词算法
    """
    label2index = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    index2label = {0: 'B', 'M': 1, 'E': 2, 'S': 3}

    def __init__(self, model):
        """
        :param model: 训练好的模型
        """
        self.model = model

    def cutsentence(self, string):
        """
        对语句进行切分
        :param string:
        :return: table
        """
        table = {}  # {‘字’:[(,),], ...}
        for i in range(len(string)):
            char = string[i]
            table[char] = [(.0, 0), (.0, 0), (.0, 0), (.0, 0)]
            if i == 0:  # 首字
                for status in ['B', 'S']:
                    table[char][self.label2index[status]] = ((self.model.model[char].post_context[status][
                        string[i + 1]] if (i + 1) < len(string)
                        and char in self.model.model.keys() and  string[i + 1] in
                        self.model.model[char].
                        post_context[status].keys() else 0) +
                        self.model.model[char].status_possible[
                        self.label2index[status]] if char in self.model.model.keys() \
                                else 0, 0)
            else:
                if i == len(string) - 1:
                    status_lst = ['E', 'S']
                else:
                    status_lst = ['B', 'M', 'E', 'S']
                for status in status_lst:
                    max_prestatus = 'S'
                    max_preval = 0.0
                    if status == 'B':
                        pstatus_lst = ['E', 'S']
                    elif status == 'M':
                        pstatus_lst = ['B', 'M']
                    elif status == 'E':
                        pstatus_lst = ['B', 'M']
                    else:
                        pstatus_lst = ['E', 'S']
                    for pstatus in pstatus_lst:
                        if string[i-1] in self.model.model.keys() and  self.model.model[string[i - 1]].transfer[self.label2index[pstatus]][
                            self.label2index[status]] * \
                                table[string[i - 1]][self.label2index[pstatus]][0] >= max_preval:
                            max_preval = self.model.model[string[i - 1]].transfer[self.label2index[pstatus]][
                                self.label2index[status]] * \
                                table[string[i - 1]][self.label2index[pstatus]][0]
                            max_prestatus = pstatus
                    table[char][self.label2index[status]] = (
                        max_preval + self.model.model[char].pre_context[status][string[i - 1]] \
                        if char in self.model.model.keys() and string[i - 1]
                        in self.model.model[
                            char].pre_context[
                            status].keys()
                        else 0 + (self.model.model[char].post_context[status][string[i + 1]] if (i + 1) < len(
                            string) and char in self.model.model.keys()  and string[i + 1] in self.model.model[char].post_context[status].keys()\
                         else 0) +
                        (self.model.model[char].status_possible[self.label2index[status]] if \
                         char in self.model.model.keys() else 0),
                        self.label2index[max_prestatus])
        # print('DEBUG:')
        # for key, val in table.items():
        #     print(key, ' ', end=' ')
        #     for a, b in val:
        #         print('{}({})'.format(a, b), end='  ')
        #     print('')
        return table


def file_cutword(srcfile, destfile, crf):
    """
    对文件进行分词
    :param srcfile:
    :param destfile:
    :param crf CRF
    :return:
    """

    with open(srcfile, encoding='utf-8', mode='r') as fr, open(destfile, encoding='utf-8', mode='w') as fw:
        process_flag = 0
        for line in fr.readlines():
            line = line.strip()
            if line == '':
                print('', file=fw)
                continue
            process_flag += 1
            if process_flag % 20 == 0:
                print('STATUS: 当前分词进度 {}行'.format(process_flag))
            table = crf.cutsentence(line)
            lasttableline_maxval = 0.0
            max_pre = 0
            max_index = 0
            try:
                laststr = line[len(line)-1]
                for i in range(4):
                    if table[laststr][i][0] >= lasttableline_maxval:
                        max_pre = table[laststr][i][1]
                        max_index = i
            except IndexError:
                print('ERROR string is:[{}]'.format(line))
            res = []
            max_pre = max_index
            i = len(line) - 1
            while i >= 0:
                if max_pre == 3:
                    res = [line[i]]+res
                    max_pre = table[line[i]][max_pre][1]
                    i -= 1
                elif max_pre == 2:
                    tmpstr = ''
                    while max_pre != 0 and i >= 0:
                        tmpstr = line[i]+tmpstr
                        max_pre = table[line[i]][max_pre][1]
                        i -= 1
                    if i >= 0 and max_pre == 0:
                        tmpstr = line[i]+tmpstr
                        max_pre = table[line[i]][max_pre][1]
                        i -= 1
                    res = [tmpstr]+res
                elif max_pre == 0 or max_pre == 1:
                    tmpstr = ''
                    while i >= 0 and max_pre != 2 and max_pre != 3:
                        tmpstr = line[i]+tmpstr
                        max_pre = table[line[i]][max_pre][1]
                        i -= 1
                    res = [tmpstr]+res
            print('  '.join(res), file=fw)


def save_model(model, file):
    """
    保存模型
    :param model:
    :param file:
    :return:
    """
    with open(file, mode='wb') as f:
        pickle.dump(model, f)


def load_model(file):
    """
    读取模型
    :param model:
    :param file:
    :return:
    """
    with open(file, mode='rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
#    model_path = pathlib.Path(MODEL_FILE)
#    if model_path.exists():
#        crf = CRF(load_model(MODEL_FILE))
#        file_cutword(TESTFILE, RESULT_FILE, crf)
#    else:
#        model = Model()
#        model.train_model(CIYU_4PLACE_LABEL_SIGHAN)
#        print('模型训练完毕。。。。')
#        save_model(model, MODEL_FILE)
#        crf = CRF(model)
#        print('文本分词中....')
#        file_cutword(TESTFILE, RESULT_FILE, crf)
    model=Model()
    model.train_model(CIYU_4PLACE_LABEL_SIGHAN)
    print('模型训练完毕.....')
    crf=CRF(model)
    print('文本分词中........')
    file_cutword(TESTFILE,RESULT_FILE,crf)
