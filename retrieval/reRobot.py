#-*-coding:utf-8-*-
from .bm25 import *
from .cut import cut
import random


class RetrievalRobot(object):


    def __init__(self, data):
        self.data = data
        self.index2question = data['origin'].to_dict()
        self.corpus = data['split'].values
        self.bm25model = self.build_bm25()

    def find_top_score_index(self, sentence):
        """寻找排序高的相似问"""
        score_overall = {}
        sentence_cut =  sentence.split(' ')#cut(sentence)
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

    def top_k(self, query_str, topk=3):
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
            if query_str == self.index2question[index]:
                indexlist.append(index)
                continue
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


