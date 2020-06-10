#-*-coding:utf-8-*-
from .bm25 import *
from .cut import cut_en,cut
import random
import gensim
import numpy as np
from annoy import AnnoyIndex


class RetrievalRobot_outsea(object):


    def __init__(self, data,word2vec_path=None):
        # 英文词向量地址
        self.word2vec_path = word2vec_path
        self.data = data
        self.index2question = data['origin'].to_dict()
        self.corpus = data['origin'].apply(lambda x: cut_en(x))
        self.bm25model = self.build_bm25()
        if self.word2vec_path is not None:
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
            self.build_vector_search(self.corpus)

    def find_top_score_index(self, sentence):
        """寻找排序高的相似问"""
        score_overall = {}
        sentence_cut =  cut_en(sentence)
        for word in sentence_cut:
            if word not in self.bm25model.document_score:
                continue
            for key, value in self.bm25model.document_score[word].items():
                if key not in score_overall:
                    # print(score_overall)
                    score_overall[key] = value
                else:
                    score_overall[key] += value
        if score_overall:
            return sorted(score_overall.items(), key=lambda x: x[1], reverse=True)
        else:
            return None

    def build_bm25(self):
        return BM25(self.corpus)

    def find_top_k_vector(self,sentence,top_k_vector):
        sentence_vector = self.encode(sentence)
        answer_index = self.vector_search_model.get_nns_by_vector(sentence_vector, top_k_vector, search_k=-1, include_distances=False)
        return answer_index

    def encode(self,sentence):
        vec = []
        for word in sentence:
            if word in self.word2vec:
                vec.append(self.word2vec[word])
        return np.mean(vec, axis=0)

    def build_vector_search(self,corpus):

        vectors = [self.encode(x) for x in corpus]
        annoy = AnnoyIndex(300, 'angular')  # Length of item vector that will be indexed
        for index, item in enumerate(vectors):
            annoy.add_item(index, item)
        annoy.build(50)
        self.vector_search_model = annoy

        # answer_index = annoy_home.get_nns_by_vector(vector, self.topk, search_k=-1, include_distances=False)
        # near_ids = []
        # for item in answer_index:
        #     near_ids.append(index2id[str(item)])


    def top_k(self, query_str, topk=3,top_k_vector=0):
        """
        搜索top k的相似问答
        zheli
        :param query_str:
        :param topk:
        :return: List [(question,answer,grades),...]
        """
        key_answers = self.find_top_score_index(query_str)
        # print('候选答案', key_answers)

        if not key_answers:
            return None
        bm25_result = []
        indexlist = []
        for index, grades in key_answers[0:topk]:
            if query_str == self.index2question[index]:  # 找出来的是自己，则not添加
                indexlist.append(index)
                continue
            bm25_result.append(self.index2question[index])#, grades 不记录得分
            indexlist.append(index)
        # 向量检索出来的索引
        if self.word2vec_path is not None:
            vector_index = self.find_top_k_vector(query_str, top_k_vector)
            for index in vector_index:
                if query_str == self.index2question[index]:  # 找出来的是自己，则not添加
                    indexlist.append(index)
                    continue
                if index in indexlist: continue
                bm25_result.append(self.index2question[index])#, grades 不记录得分
                indexlist.append(index)

        if len(bm25_result) < topk:
            # 随机选取进行补充
            temp_question_index = []
            for i in self.index2question.keys():
                if i not in indexlist:
                    temp_question_index.append(self.index2question[i])
            random.shuffle(temp_question_index)
            random_result = temp_question_index[:(topk-len(bm25_result))]

            result = bm25_result+random_result

            if len(result) < topk:
                print('topK设置过大，超过候选问题总数')
            return result
        else:
            result = bm25_result
            return result


