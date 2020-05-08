import spacy
import csv
import re
import ray
import multiprocessing
from functools import partial
from tqdm import tqdm
from itertools import chain
from random import random, shuffle, randint
from collections import Counter
from spacy.symbols import nsubj, VERB, PRON, root

DELIMS = {
    'section': '~',
    'entity_relation': '`',
    'keywords': '^',
    'title': '@',
    'body': '}'
}

PRONOUN_LIST = ['I', 'Me', 'We', 'You', 'He', 'She',
                'It', 'Him', 'Her', 'Them', 'They']

PRONOUNS = set(PRONOUN_LIST + [x.lower() for x in PRONOUN_LIST])


def encode_keywords(csv_path, model='en_core_web_sm',
                    entity_relation_field=None,
                    keywords_field=None,
                    title_field=None,
                    body_field=None,
                    keyword_gen='title',
                    keyword_sep=',',
                    dropout=0.5,
                    repeat=3,
                    max_keywords=10,
                    min_keywords=5,
                    keyword_length_max=20,
                    out_path='csv_encoded.txt',
                    start_token="<|startoftext|>",
                    end_token="<|endoftext|>"):

    data_list = []

    if max_keywords < min_keywords:
        raise Exception('max_keywords cannot be less than min_keywords')

    with open(csv_path, 'r', encoding='utf8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_list.append(row)

    shuffle(data_list)

    # https://stackoverflow.com/a/434328
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    num_threads = multiprocessing.cpu_count() * 2  # colocate 2 processes per thread
    print("Starting up {} Workers".format(num_threads))
    encoders = [Encoder.remote(model, entity_relation_field,
                               keywords_field,
                               title_field,
                               body_field,
                               keyword_gen,
                               keyword_sep,
                               repeat,
                               max_keywords,
                               min_keywords,
                               keyword_length_max,
                               start_token,
                               end_token,
                               DELIMS,
                               PRONOUNS) for _ in range(num_threads)]

    with open(out_path, 'w', encoding='utf8', errors='ignore') as w:
        # Instantly make your loops show a smart progress meter
        pbar = tqdm(total=len(data_list), smoothing=0)
        for chunk in chunker(data_list, num_threads):
            results = ray.get([c.generate_encoded_text.remote(row)
                               for c, row in list(zip(encoders, chunk))])

            # unnest and randomize results
            results = list(chain.from_iterable(results))
            shuffle(results)
            for result in results:
                w.write(result)

            pbar.update(num_threads)
        pbar.close()


@ray.remote(num_cpus=0.5)
class Encoder(object):
    def __init__(self, model, entity_relation_field,
                 keywords_field,
                 title_field,
                 body_field,
                 keyword_gen,
                 keyword_sep,
                 repeat,
                 max_keywords,
                 min_keywords,
                 keyword_length_max,
                 start_token,
                 end_token,
                 DELIMS,
                 PRONOUNS):
        self.nlp = spacy.load(model)
        self.pattern = re.compile('\W+')

        self.entity_relation_field = entity_relation_field
        self.keywords_field = keywords_field
        self.title_field = title_field
        self.body_field = body_field
        self.keyword_gen = keyword_gen
        self.keyword_sep = keyword_sep
        self.repeat = repeat
        self.max_keywords = max_keywords
        self.min_keywords = min_keywords
        self.keyword_length_max = keyword_length_max
        self.start_token = start_token
        self.end_token = end_token
        self.DELIMS = DELIMS
        self.PRONOUNS = PRONOUNS

    def build_section(self, section, text):
        if text is None:
            return ''
        return self.DELIMS['section'] + self.DELIMS[section] + text

    def generate_encoded_text(self, row):

        nlp = self.nlp
        pattern = self.pattern
        # replace smart quotes first for better tokenization
        text = re.sub(u'[\u2018\u2019]', "'",
                      (re.sub(u'[\u201c\u201d]', '"', row[self.keyword_gen])))
        doc = nlp(text)

        # category should be normalized to account for user input
        # category = re.sub(
        #     pattern, '-', row[self.category_field].lower().strip()) if self.category_field is not None else None

        # entity_relation, e.g. {relation: [entity1, entity2]}
        # relation is normally verb
        entity_dependency_labels = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'obj', 'dobj', 'pobj']
        modifier_labels = ['advmod', 'amod', 'appos', 'meta', 'nn']
        relation_labels = ['ROOT', 'prep-x', 'agent', 'attr', 'cc', 'conj'] + modifier_labels
        entity_relation_pairs = {}
        for possible_entity in doc:
            if possible_entity.pos == PRON:
                continue
            if possible_entity.dep_ in entity_dependency_labels and possible_entity.head.dep_ in relation_labels:
                key = entity_relation_pairs.get(possible_entity.head.lemma_)
                if not key:
                    entity_relation_pairs[possible_entity.head.lemma_] = {possible_entity.lemma_}
                else:
                    key.add(possible_entity.lemma_)

        title = row[self.title_field] if self.title_field is not None else None
        body = row[self.body_field] if self.body_field is not None else None

        if self.keywords_field is None:
            # Generate the keywords using spacy
            keywords_pos = [chunk.text if chunk.pos_ == 'NOUN'
                            else chunk.lemma_ if chunk.pos_ in ['VERB', 'ADJ', 'ADV']
                            else 'I'
                            for chunk in doc
                            if not chunk.is_stop
                            ]
            keywords_ents = [re.sub(' ', '-', chunk.text)
                             for chunk in doc.ents]
            keywords_compounds = [re.sub(' ', '-', chunk.text)
                                  for chunk in doc.noun_chunks
                                  if len(chunk.text) < self.keyword_length_max]

            # Origin version
            # keywords = list(set(keywords_pos +
            #                     keywords_ents +
            #                     keywords_compounds) - self.PRONOUNS)  # dedupe

            hotwords = list(set([x[0] for x in Counter(keywords_pos +
                                                       keywords_ents +
                                                       keywords_compounds).most_common()
                            if x[1] > 1]) - self.PRONOUNS)
            keywords = list(set(keywords_pos +
                                keywords_ents) - self.PRONOUNS - set(hotwords))
        else:
            keywords = [keyword.strip()
                        for keyword in row[self.keywords_field].split(self.keyword_sep)]
            keywords = list(set(keywords))
            hotwords = []

        encoded_texts = []
        for _ in range(self.repeat):
            shuffle(keywords)
            if self.max_keywords <= len(hotwords):
                shuffle(hotwords)
                new_keywords = hotwords[:randint(self.min_keywords, self.max_keywords)]
            elif self.min_keywords <= len(hotwords) < self.max_keywords:
                new_keywords = hotwords + keywords[:randint(0, self.max_keywords - len(hotwords))]
            else:
                # len(hotwords) < self.min_keywords
                new_keywords = hotwords + keywords[:randint(self.min_keywords - len(hotwords),
                                                            self.max_keywords - len(hotwords))]

            shuffle(new_keywords)
            new_keywords = " ".join(new_keywords)

            encoded_texts.append(self.start_token +
                                 self.build_section('entity_relation',
                                                    str(entity_relation_pairs).strip('{}').
                                                    replace("'", '').replace('{', '(').replace('}', ')') + ')') +
                                 self.build_section('keywords', new_keywords) +
                                 self.build_section('title', title) +
                                 self.build_section('body', body) +
                                 self.end_token + "\n")
        return encoded_texts
