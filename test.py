from NaiveBayes import *
import numpy as np
import re
import jieba
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from collections import defaultdict
def get_corpus(path,train=True):
    if train:
        with open("train_data/sample.positive.txt", 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        a = re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)
        pos_len = len(a)
        with open("train_data/sample.negative.txt", 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')    
        a += re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)
    else:
        with open("train_data/sample.postive.txt", 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        a = re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)
        pos_len = len(a)
    stop_words = set()
    with open("stopwords.txt","r",encoding="utf-8") as s:
        for line in s.readlines():
            line = line.strip()
            stop_words.add(line)
    split_corpus = [[word for word in jieba.lcut(x) if word not in stop_words] for x in a]
    frequency = defaultdict(int)
    for line in split_corpus:
        for token in line:
            frequency[token] += 1
    split_corpus = [[word for word in x if frequency[word] >= 5] for x in split_corpus]
    word_dict = Dictionary(split_corpus)
    word_index = {word:index for index, word in word_dict.items()}
    print(len(word_index))
    feature_matrix = np.zeros((len(a),len(word_index)+1),dtype=int)#特征向量，最后一行是分类标签
    for i in range(len(a)):
        for word in split_corpus[i]:
            feature_matrix[i,word_index[word]] = 1
    if train:
        feature_matrix[0:pos_len,-1] = 1
    return feature_matrix
def train_test():
    nb = NaiveBayes(0)
    get_corpus("train")
    nb.trainer(np.loadtxt("train_data/train.txt"))
    get_corpus("test")
    test = np.loadtxt("test_data/test.txt")
    res = nb.predictor(test)
def k_():
    nb = NaiveBayes()
    all_data = get_corpus("train")
    train_x, test_x, train_y, test_y = train_test_split(all_data[:,:-1], all_data[:,-1], test_size=0.2)
    train_y = np.reshape(train_y,(len(train_y),1))
    nb.trainer(np.hstack((train_x,train_y)))
    test_y = np.reshape(test_y,(len(test_y),1))
    test = np.hstack((test_x,test_y))
    res = nb.predictor(test)
    print(test_y.shape)
    count = 0
    for i in range(test_y.shape[0]):
        count = count+1 if test_y[i][0]==res[i] else count
    print("test accuracy:"+str(count/test_y.shape[0]))
if __name__ == "__main__":
    k_()
