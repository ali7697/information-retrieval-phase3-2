import heapq
import math
import pickle
from copy import deepcopy


class Dictionary:

    def __init__(self):
        self.dictionary = dict()
        self.words_total_count = dict()
        self.id_to_url_dict = dict()
        self.doc_vectors = dict()
        self.doc_vectors_lengths = dict()
        self.champions_list_dict = dict()
        self.cluster_dict = dict()
        self.sport_dict = dict()
        self.political_dict = dict()
        self.economy_dict = dict()
        self.health_dict = dict()
        self.culture_dict = dict()
        self.needed_to_be_saved_dicts_names = ['dict', 'dict_url', 'docs_vectors', 'doc_vector_lengths',
                                               'sport', 'political', 'economy', 'health', 'culture']
        self.needed_to_be_saved_dicts_dictionaries = [self.dictionary, self.id_to_url_dict,
                                                      self.doc_vectors, self.doc_vectors_lengths,
                                                      self.sport_dict, self.political_dict,
                                                      self.economy_dict, self.health_dict,
                                                      self.culture_dict]

    def sort_dict(self):
        # create the sorted final dictionary
        words = sorted(self.dictionary.keys())
        final_dictionary = dict()
        for word in words:
            final_dictionary[word] = self.dictionary[word]
        self.dictionary = final_dictionary

    def save_dict(self):
        for i in range(len(self.needed_to_be_saved_dicts_dictionaries)):
            filename = self.needed_to_be_saved_dicts_names[i]
            outfile = open(filename, 'wb')
            pickle.dump(self.needed_to_be_saved_dicts_dictionaries[i], outfile)
            outfile.close()

    def read_dict(self):
        dicts = []
        for i in range(len(self.needed_to_be_saved_dicts_dictionaries)):
            infile = open(self.needed_to_be_saved_dicts_names[i], 'rb')
            dicts.append(pickle.load(infile))
            infile.close()
        return dicts

    def test_dict(self, input_word):
        if input_word in self.dictionary.keys():
            print(self.dictionary[input_word])
        else:
            print("This word is not in the dictionary")

    def get_frequency_based_dict(self):
        freq_based_dict = sorted(self.words_total_count.items(), key=lambda x: x[1], reverse=True)
        # freq_based_dict = dict(freq_based_dict)
        return freq_based_dict

    def remove_k_frequent_words(self, k):
        words = self.get_frequency_based_dict()[0:k]
        for word in words:
            del self.dictionary[word[0]]

    def create_doc_vectors(self):
        for ID in self.id_to_url_dict.keys():
            self.doc_vectors[ID] = dict()
            self.doc_vectors[ID]['cluster'] = self.cluster_dict[ID]
            # self.doc_vectors[ID]['cluster'] =
        for word in self.dictionary.keys():
            for i in range(1, len(self.dictionary[word])):
                self.doc_vectors[self.dictionary[word][i][0]][word] = self.dictionary[word][i][1]

    def cal_doc_vectors_lengths(self):
        for x in self.doc_vectors.keys():
            self.doc_vectors_lengths[x] = math.sqrt(sum([j ** 2 for j in list(self.doc_vectors[x].values())[1:]]))

    def create_champions_list(self):
        tmp_dict = deepcopy(self.dictionary)
        for word in self.dictionary.keys():
            tmp_dict[word].remove(tmp_dict[word][0])
            largest = heapq.nlargest(40, tmp_dict[word], key=lambda x: x[1])
            largest = sorted(largest, key=lambda x: x[0])
            largest.insert(0, self.dictionary[word][0])
            self.champions_list_dict[word] = largest
        return self.champions_list_dict
