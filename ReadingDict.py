from Retriver import *
from score import *


dictionary = Dictionary()
dicts = dictionary.read_dict()
dictionary.dictionary = dicts[0]
dictionary.id_to_url_dict = dicts[1]
dictionary.doc_vectors = dicts[2]
dictionary.doc_vectors_lengths = dicts[3]
dictionary.sport_dict = dicts[4]
dictionary.political_dict = dicts[5]
dictionary.economy_dict = dicts[6]
dictionary.health_dict = dicts[7]
dictionary.culture_dict = dicts[8]



print(len(dictionary.dictionary.keys()))
ret = Retriever(dictionary)
ret.get_query()
ret.retrieve()

