import math
import random
from copy import deepcopy
import heapq
from Equalizer import *
from score import *
from knn_ing import *
from clusters import *
import numpy as np
from sklearn.model_selection import KFold


class Retriever:
    query: str

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.equalizer = Equalizer(self.dictionary)
        self.initial_doc_vectors = deepcopy(self.dictionary.doc_vectors)
        self.query_category = ""

    def get_dict(self):
        dicts = self.dictionary.read_dict()
        self.dictionary.dictionary = dicts[0]
        self.dictionary.id_to_url_dict = dicts[1]
        self.dictionary.doc_vectors = dicts[2]
        self.dictionary.doc_vectors_lengths = dicts[3]
        self.dictionary.sport_dict = dicts[4]
        self.dictionary.political_dict = dicts[5]
        self.dictionary.economy_dict = dicts[6]
        self.dictionary.health_dict = dicts[7]
        self.dictionary.culture_dict = dicts[8]

    def get_query(self):
        self.query = input('Please enter the query: ')
        inp = self.query.split(" ")
        self.query_category = inp[0]
        self.query = ''
        for element in inp[1:]:
            self.query += element + ' '
    def get_equalized_query(self):
        alphabet = 'آاأبپتثجچحخدذرزژسشصضطظعغفقکكگلم‌نوؤهیيئء'  # نیم فاصله داره
        r = re.compile(f'[{alphabet}]+')
        words = r.findall(self.query)
        words = self.equalizer.equalize_query(words)
        return words

    def cal_cosine_similarity(self, main_doc, cluster):
        doc_similarity_dict = dict()
        main_doc_vector_length = math.sqrt(sum([j ** 2 for j in list(main_doc.values())]))
        for doc_id in cluster.doc_vector_dict.keys():
            if self.dictionary.doc_vectors_lengths[doc_id] == 0:
                continue
            tmp_sum = 0
            for term in main_doc.keys():
                if term in cluster.doc_vector_dict[doc_id].keys() and term != 'cluster':
                    tmp_sum += main_doc[term] * cluster.doc_vector_dict[doc_id][term]
            tmp_sum /= (cluster.length[doc_id] * main_doc_vector_length)
            doc_similarity_dict[doc_id] = tmp_sum
        return doc_similarity_dict

    def retrieve(self):
        words = self.get_equalized_query()
        # create query vector
        query_vector = dict()
        for word in words:
            query_vector[word] = 1 + math.log10(words.count(word))
        cluster = Cluster(-3, self.dictionary, 0)
        if self.query_category == 'sport':
            cluster.doc_vector_dict = self.dictionary.sport_dict
        else:
            if self.query_category == 'political':
                cluster.doc_vector_dict = self.dictionary.political_dict
            else:
                if self.query_category == 'economy':
                    cluster.doc_vector_dict = self.dictionary.economy_dict
                else:
                    if self.query_category == 'health':
                        cluster.doc_vector_dict = self.dictionary.health_dict
                    else:
                        cluster.doc_vector_dict = self.dictionary.culture_dict
        cluster.cal_length()
        doc_similarities = self.cal_cosine_similarity(query_vector, cluster)
        largest = heapq.nlargest(30, list(doc_similarities.items()), key=lambda x: x[1])
        largest = sorted(largest, key=lambda x: x[1], reverse=True)
        output_printed = dict()
        print('Num of docs: ' + str(len(cluster.doc_vector_dict.keys())))
        for pair in largest:
            output_printed[pair[0]] = self.dictionary.id_to_url_dict[pair[0]]
            print(str(pair[0]) + '\t' + output_printed[pair[0]])
        return output_printed
