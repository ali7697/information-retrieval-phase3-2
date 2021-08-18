from News import *
from Retriver import *
from score import score
from Dictionary import *

dictionary = Dictionary()
dictionary_3 = Dictionary()

All_News = []
equalizer = Equalizer(dictionary)
equalizer_3 = Equalizer(dictionary_3)
# read the News file
with open('IR_Spring2021_ph12_7k.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        # if line == 500:
        #     break
        if line != 0:
            n = News(row[0], row[1], row[2], equalizer, dictionary)
            All_News.append(n)
            d = n.abc()
        line += 1
dictionary.remove_k_frequent_words(50)
equalizer.equalize_dict()
# dictionary1 = equalizer.ret_dict()
dictionary.sort_dict()
score_giver = score(dictionary)
score_giver.cal_tf()
score_giver.cal_idf()
score_giver.cal_tfidf()
dictionary.create_doc_vectors()
dictionary.cal_doc_vectors_lengths()

id_counter = 0
with open('17k.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer, dictionary_3, row[2])
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

with open('20k.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer, dictionary_3, row[2])
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

with open('IR00_3_11k News.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
        if line != 0:
            n = News(id_counter, row[1], row[3], equalizer, dictionary_3, row[2])
            All_News.append(n)
            d = n.abc()
            id_counter += 1
        line += 1

dictionary_3.remove_k_frequent_words(50)
equalizer_3.equalize_dict()
dictionary_3.sort_dict()
score_giver = score(dictionary_3)
score_giver.cal_tf()
score_giver.cal_idf()
score_giver.cal_tfidf()
dictionary_3.create_doc_vectors()
dictionary_3.cal_doc_vectors_lengths()

print('done with dicts')

knn = KNN(dictionary, dictionary_3)
knn.find_cluster()

dictionary.save_dict()
ret = Retriever(dictionary)
ret.get_query()
ret.retrieve()
