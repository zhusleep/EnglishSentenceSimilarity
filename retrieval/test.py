from copy import deepcopy
import pandas as pd
from retrieval.reRobot_outsea import RetrievalRobot_outsea
from retrieval.cut import cut_en
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

standard_df = pd.read_csv("../data/standard.csv" , sep="\t")
standard_df.drop_duplicates(subset=['standard'], keep='first', inplace=True)
standard_df['split'] = standard_df.standard.apply(lambda x: cut_en(x))
standard_df['origin'] = standard_df.standard
reRobot_std = RetrievalRobot_outsea(standard_df,word2vec_path='../GoogleNews-vectors-negative300.bin')
text = 'may i have a question'

result = reRobot_std.top_k(text, 5, top_k_vector=0)
print(result)

