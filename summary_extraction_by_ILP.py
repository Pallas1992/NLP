# -*- coding: utf-8 -*-
"""
@auther: Tie Ruixue
@file: ILP.py
@time: 2017/12/17 18:12
"""
import nltk
import math
import numpy as np
from pymprog import *

doc = "I come from China, and I am a girl. I'd like to make friends.\
My work concerns to  Natural Language Processing. It is a tough job for me.\
I want to find some courses for my work.Oh,God. I feel it is hard."
#-------文档预处理、切分为句子----------
def get_sent(text):
    #sentences = text1.split('.')
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        if len(sentence)<=3:
            sentences.remove(sentence)
    return sentences
def get_basic_list(text):
    words_doc = nltk.word_tokenize(text)
    basic_list = sorted(set(words_doc), key = words_doc.index)
    return basic_list
# -------计算参数-----------------------
#句子向量化
def get_sent_vector(text,sentences):
    sen_vector = []
    for sentence in sentences:
        n = []
        basic_list = get_basic_list(text)
        for word in basic_list:
            if word in sentences:
                n.append(sentence.count(word))
            else:
                n.append(0)
        sen_vector.append(n)
    return sen_vector
#文档向量化
def get_doc_vector(text):
    doc_vector = []
    basic_list = get_basic_list(text)
    for w in basic_list:
        doc_vector.append(text.count(w))
    return doc_vector
#每个句子与文档的相关性
def cos_sd(sen_vector, doc_vector):
    doc_mat = np.mat(doc_vector)
    norm_doc_mat = doc_mat * doc_mat.T
    sen_corr_doc = []
    for v in sen_vector:
        sen_mat = np.mat(v)
        norm_sen_mat = np.dot(sen_mat, sen_mat.T)
        if (math.sqrt(norm_sen_mat * norm_doc_mat ) == 0):
            cosine_sd = 0
        else:
            cosine_sd = np.dot(sen_mat, doc_mat.T) / math.sqrt(norm_sen_mat * norm_doc_mat)
        sen_corr_doc.append(cosine_sd)
    return sen_corr_doc
#每两个句子之间的相关性
def cos_ss(sen_vector):
    sen_corr_sen = []
    for i in sen_vector:
        a = []
        seni_mat = np.mat(i)
        norm_seni_mat = np.dot(seni_mat, seni_mat.T)
        for j in sen_vector:
            senj_mat = np.math(j)
            norm_senj_mat = np.dot(senj_mat, senj_mat.T)
            if i != j:
                if (math.sqrt(norm_seni_mat * norm_senj_mat) == 0):
                    cosine_ss = 0
                else:
                    cosine_ss = np.dot(seni_mat, senj_mat.T) / math.sqrt(norm_seni_mat * norm_senj_mat)
            else:
                cosine_ss = 1.0
        sen_corr_sen.append(a)
    return sen_corr_sen
#每个句子在文档中位置的倒数
def get_sen_pos(sentences):
    sen_pos = []
    for sentence in sentences:
        sen_index = 1.0/(sentences.Index(sentence) + 1)
        sen_pos.append(sen_index)
    return sen_pos
#每个句子去嵌套
def l_flatten(list):
    a = []
    for i in list:
        for ii in i:
            a.append(ii)
    return a

if __name__ == '__main__':
    sentences = get_sent(doc)
    sen_vector = get_sent_vector(doc, sentences)
    doc_vector = get_doc_vector(doc)
    sen_corr_sen = cos_sd(sen_vector,doc_vector)
    pos = get_sen_pos(sentences)
    Reli = []
    for i in range(len(np.array(pos))):
        d = np.array(sen_corr_doc)[i][0,0] + np.array(pos)[i]
        Reli.append(d)
    print('------Reli:------------')
    print(Reli)
    Redij1 = cos_ss(sen_vector)
    Redij = []
    for i in Redij1:
        b = []
        for j in i:
            if isinstance(j, np.matrix):
                b.append(j.tolist()[0][0])
            else:
                b.append(j)
        Redij.append(b)
    print('------Redij:------------')
    print(Redij)
    # 计算公式
    L = 20 #摘要句子的长度
    Lj = [] #文章中每个句子的长度
    for j in sentences:
        lj = len(j.split())
        Lj.append(lj)

    begin('ILP')
    Si = var('Si',len(sentences),kind = bool)
    Sj = var('Sj', len(sentences), kind=bool)
    Reli = par('Reli', Reli)
    Redij = par('Redij', Redij)
    Sij =  np.array(np.mat(Si).T * np.mat(Sj))
    #也可以用下面的方法
    # T = list(iprod(Si, Sj))
    # Sij = var('Sij', len(T), bool)
    s_1 = np.array(np.mat(Si).T * (np.mat(Sj) - 1))
    s_2 = np.array((np.mat(Si) - 1).T * np.mat(Sj))
    s_3 = np.array(np.mat(np.array(Si)).T) - s_2

    maximize((sum(p*q for p,q in zip(Reli,Si)) - sum(map(sum, np.array(Redij) * Sij))), 'abstract_value')
    st(sum(p * q for p,q in zip(Lj,Sj)) <= L)
    st(p <= 0 for p in l_flatten(s_1))
    st(q <= 0 for q in l_flatten(s_2))
    st(k <= 1 for k in l_flatten(s_3))

    solve()
    sensitivity()




