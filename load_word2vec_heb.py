from gensim.models.word2vec import Word2Vec

word2vec = Word2Vec.load("./wiki.word2vec.model")  # the model file is next to this script at github - https://github.com/lisrael1/word2vec-heb-wiki/blob/main/wiki.word2vec.model
word2vec.wv.most_similar('משחקים', topn=5)
