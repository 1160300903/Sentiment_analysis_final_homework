import re
import jieba
import numpy as np
from keras.preprocessing import sequence
from gensim.models import word2vec
from keras.utils import to_categorical
from gensim.corpora.dictionary import Dictionary
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.core import Activation
from sklearn.model_selection import train_test_split
import yaml
import argparse
def get_corpus():
    with open("train_data/sample.positive.txt", 'r', encoding='utf-8')as f:
        text = f.read().replace('\n', '')
    a = re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)
    pos_len = len(a)
    with open("train_data/sample.negative.txt", 'r', encoding='utf-8')as f:
        text = f.read().replace('\n', '')    
    a += re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)
    neg_len = len(a)-pos_len
    stop_words = set()
    with open("stopwords.txt","r",encoding="utf-8") as s:
        for line in s.readlines():
            line = line.strip()
            stop_words.add(line)
    split_corpus = [[word for word in jieba.lcut(x) if word not in stop_words] for x in a]
    split_corpus = np.array(split_corpus)
    model  = word2vec.Word2Vec(split_corpus,size = 300,min_count =5,window =3)
    print(split_corpus.shape)
    tag = np.concatenate((np.ones(pos_len), np.zeros(neg_len)))
    print(tag.shape)
    return split_corpus,tag,model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='binary sentiment analysis')
    parser.add_argument("-v","--vector_length",type=int,default=300,help="the length of word embeddings")
    parser.add_argument("-s","--sentence_length",type=int,default=50,help="the length of a sentence")
    args = parser.parse_args()
    embedding_length = args.vector_length
    sentence_length = args.sentence_length

    corpus, tag, model = get_corpus()
    word_dict = Dictionary()
    word_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    word_index = {word:index+1 for index, word in word_dict.items()}
    index_form_corpus = []
    for sentence in corpus:
        index_form_sentence = []
        for word in sentence:
            if word in word_index:
                index_form_sentence.append(word_index[word])
            else:
                index_form_sentence.append(0)
        index_form_corpus.append(index_form_sentence)
    index_form_corpus = sequence.pad_sequences(index_form_corpus,maxlen=sentence_length)
    index_vectors_matrix = np.zeros((len(word_index)+1,embedding_length))
    for word,index in word_index.items():
        index_vectors_matrix[index,:] = model.wv[word]
    lstm_model = Sequential()
    lstm_model.add(Embedding(output_dim=embedding_length,
    input_dim=len(word_index)+1,
    mask_zero=True,
    weights = [index_vectors_matrix],
    input_length=sentence_length))
    lstm_model.add(LSTM(activation="sigmoid",units=50))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1))
    lstm_model.add(Activation("sigmoid"))
    print("begin training")
    lstm_model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    #分割数据集
    train_x, test_x, train_y, test_y = train_test_split(index_form_corpus, tag, test_size=0.2)
    lstm_model.fit(train_x,train_y,batch_size=32,epochs=4,verbose=1)
    result = lstm_model.to_yaml()
    with open("lstm.yml","w") as output:
        output.write(yaml.dump(result,default_flow_style=True))
    lstm_model.save_weights("weights.h5")
    accuracy = lstm_model.evaluate(test_x, test_y, batch_size=32)
    print("accuracy:",accuracy) 
