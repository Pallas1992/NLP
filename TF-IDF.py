# -*- coding: utf-8 -*-

"""
@auther: Tie Ruixue
@file: TF-IDF.py
@time: 2017/12/17 15:47
"""
import nltk
import math
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords

text1 = "I come from China, and I am a girl. I'd like to make friends."
text2 = "My work concerns to  Natural Language Processing. It is a tough job for me."
text3 = "I want to find some courses for my work. But it is hard."


# ------------文档预处理---------------------
# 去标点符号，分词，去停用词，词干提取
def get_tokens(text, stemmer):
    no_punctuation = text.replace(string.punctuation, '')
    tokens = nltk.word_tokenize(no_punctuation)
    #filtered = [w for w in tokens if w not in stopwords.words('english')]
    #stemmed = [stemmer.stem(item) for item in filtered]
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed


# ------------TF-IDF---------------------
def tf_idf(word, doc, doclist):
    tf =doc.count(word)/len(doc)
    n_container = sum(1 for doc in doclist if word in doc)
    idf = math.log(len(doclist)/(1+n_container))
    tf_idf = tf*idf
    return  tf_idf
if __name__ == '__main__':
    stemmer = PorterStemmer()
    stemmed = get_tokens(text1, stemmer)
    count1 = nltk.Counter(stemmed)
    print(count1.most_common(10))
    print('----处理成功,引如TF-IDF算法-----')
    doclist = [get_tokens(text1),get_tokens(text2),get_tokens(text3)]
    for doc in doclist:
        for word in doc:
            scores = {word : tf_idf(word,doc,doclist)}
            print(scores)
