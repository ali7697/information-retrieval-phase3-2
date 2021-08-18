# from Dictionary import *
from Equalizer import *


long_words = []
short_words = []
with open('short_forms.csv', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        short_words.append(row[0])
        long_words.append(row[1])


class News:
    def __init__(self, id, content, url, equalizer, dictionary, cluster=None):
        self.tokens = []
        self.id = id
        self.content = content
        self.url = url
        self.equalizer = equalizer
        self.dictionary = dictionary
        self.cluster = cluster if cluster is not None else 0

    def abc(self):  # tokeniz content
        # add the id and url to its dict
        self.dictionary.cluster_dict[int(self.id)] = self.cluster
        self.dictionary.id_to_url_dict[int(self.id)] = self.url
        corrected_content = ''
        # added half space u200c\
        alphabet = 'آاأبپتثجچحخدذرزژسشصضطظعغفقکكگلم‌نوؤهیيئء'  # نیم فاصله داره
        r = re.compile(f'[{alphabet}]+')
        words_in_content = r.findall(self.content)
        num_token_in_content = dict()
        for token in words_in_content:
            if '\u200c' in token:
                token = re.sub('\u200c', '', token)
            token = self.equalizer.char_equalizer(token, 'q')
            corrected_content += token + ' '
            if token not in self.tokens:
                self.tokens.append(token)
                num_token_in_content[token] = 1
            else:
                num_token_in_content[token] += 1
        for token in self.tokens:
            if token in self.dictionary.dictionary.keys():
                self.dictionary.dictionary[token].append([int(self.id), num_token_in_content[token]])
                self.dictionary.words_total_count[token] += num_token_in_content[token]
            else:
                self.dictionary.dictionary[token] = [[int(self.id), num_token_in_content[token]]]
                self.dictionary.words_total_count[token] = num_token_in_content[token]
        return corrected_content

