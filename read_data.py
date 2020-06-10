import  gensim
word2vec_path = "~/Downloads/GoogleNews-vectors-negative300.bin.gz"
word2vec_path = 'GoogleNews-vectors-negative300.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
print(word2vec)
i = 1