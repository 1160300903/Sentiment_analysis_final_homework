from gensim.models import word2vec
entences = word2vec.LineSentence("")
for i in sentences:
    print(i)
    break
model  = word2vec.Word2Vec(sentences,size = 300,min_count =5,window =3)
#print(model.wv.index2word())
model.save("word2Vec.model")
model.wv.save_word2vec_format('word2Vec.model.txt', binary=False)
