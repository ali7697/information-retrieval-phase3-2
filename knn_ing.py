import random
from random import *

import numpy as np
from sklearn.model_selection import KFold

from Dictionary import *
from clusters import *

class KNN:

    def __init__(self, dictionary, dictionary_3):
        self.dictionary = dictionary
        self.dictionary_3 = dictionary_3

    def cal_cosine_similarity(self, main_doc, cluster):
        doc_similarity_dict = dict()
        main_doc_vector_length = math.sqrt(sum([j ** 2 for j in list(main_doc.values())]))
        for doc_id in cluster.doc_vector_dict.keys():
            if self.dictionary_3.doc_vectors_lengths[doc_id] == 0:
                continue
            tmp_sum = 0
            for term in main_doc.keys():
                if term in cluster.doc_vector_dict[doc_id].keys() and term != 'cluster':
                    tmp_sum += main_doc[term] * cluster.doc_vector_dict[doc_id][term]
            tmp_sum /= (cluster.length[doc_id] * main_doc_vector_length)
            doc_similarity_dict[doc_id] = tmp_sum
        return doc_similarity_dict

    def predict_doc_label(self, doc, docs_to_compare, k):
        tmp_cluster = Cluster(-2, self.dictionary_3, 0)
        tmp_cluster.doc_vector_dict = docs_to_compare
        tmp_cluster.cal_length()
        doc_similarities_dict = self.cal_cosine_similarity(doc, tmp_cluster)
        largest = heapq.nlargest(k, list(doc_similarities_dict.items()), key=lambda x: x[1])
        labels = []
        for pair in largest:
            labels.append(docs_to_compare[pair[0]]['cluster'])
        label = max(set(labels), key=labels.count)
        return label

    def predict_document_sections(self):
        k = 5
        ids = random.sample(list(np.array(list(self.dictionary_3.doc_vectors.keys()))), 1000)
        kf = KFold(n_splits=10, shuffle=True)
        accuracies = []
        for train_index, val_index in kf.split(ids):
            training_set = {ids[key]: self.dictionary_3.doc_vectors[ids[key]] for key in train_index}
            validation_set = {ids[key]: self.dictionary_3.doc_vectors[ids[key]] for key in val_index}
            correct_labels = 0
            for doc_id in validation_set.keys():
                if self.dictionary_3.doc_vectors_lengths[doc_id] == 0:
                    continue
                label = self.predict_doc_label(validation_set[doc_id], training_set, k)
                if label == validation_set[doc_id]['cluster']:
                    correct_labels += 1
            accuracies.append(correct_labels/len(validation_set))
        print(accuracies)
        print(sum(accuracies)/len(accuracies))

    def find_cluster(self):
        for doc_id in self.dictionary.doc_vectors.keys():
            # what happens if a doc is close to these
            if self.dictionary.doc_vectors_lengths[doc_id] == 0:
                continue
            label = self.predict_doc_label(self.dictionary.doc_vectors[doc_id], self.dictionary_3.doc_vectors, 5)
            if label == 'sport':
                self.dictionary.sport_dict[doc_id] = self.dictionary.doc_vectors[doc_id]
            else:
                if label == 'political':
                    self.dictionary.political_dict[doc_id] = self.dictionary.doc_vectors[doc_id]
                else:
                    if label == 'economy':
                        self.dictionary.economy_dict[doc_id] = self.dictionary.doc_vectors[doc_id]
                    else:
                        if label == 'health':
                            self.dictionary.health_dict[doc_id] = self.dictionary.doc_vectors[doc_id]
                        else:
                            self.dictionary.culture_dict[doc_id] = self.dictionary.doc_vectors[doc_id]