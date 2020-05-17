import spacy
import csv
import re
import ray
import multiprocessing
import textacy
import neuralcoref
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
                    enable_keyword=True,
                    enable_svo=True,
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
                               enable_keyword,
                               enable_svo,
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
                 enable_keyword,
                 enable_svo,
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
        self.enable_keyword = enable_keyword
        self.enable_svo = enable_svo
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

    @staticmethod
    def extract_subject_verb_object_triples(doc):
        lemma_triples = []
        triples = textacy.extract.subject_verb_object_triples(doc)
        for triple in triples:
            _subject, _verb, _object = triple
            _subject = _subject.root
            _object = _object.root
            _verb = _verb.root.lemma_
            if _subject._.in_coref:
                _subject = _subject._.coref_clusters[0].main.text
            if _object._.in_coref:
                _object = _object._.coref_clusters[0].main.text
            lemma_triples.append((_subject, _verb, _object))
        return lemma_triples

    def extract_keywords(self, doc):
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

        hotwords = list(set([x[0] for x in Counter(keywords_pos +
                                                   keywords_ents +
                                                   keywords_compounds).most_common()
                        if x[1] > 1]) - self.PRONOUNS)
        keywords = list(set(keywords_pos +
                            keywords_ents) - self.PRONOUNS - set(hotwords))
        return keywords, hotwords

    def shuffle_keywords(self, keywords, hotwords):
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
        return new_keywords

    def generate_encoded_text(self, row):

        nlp = self.nlp
        if not nlp.has_pipe('neuralcoref'):
            neuralcoref.add_to_pipe(nlp)
        pattern = self.pattern
        # replace smart quotes first for better tokenization
        text = re.sub(u'[\u2018\u2019]', "'",
                      (re.sub(u'[\u201c\u201d]', '"', row[self.keyword_gen])))
        doc = nlp(text)

        # triples of (subject, verb, object)
        svo_triples = []
        if self.enable_svo:
            svo_triples = self.extract_subject_verb_object_triples(doc)

        title = row[self.title_field] if self.title_field is not None else None
        body = row[self.body_field] if self.body_field is not None else None
        keywords = []
        hotwords = []
        if self.enable_keyword:
            keywords, hotwords = self.extract_keywords(doc)

        encoded_texts = []
        if not self.enable_keyword:
            self.repeat = 1
        for _ in range(self.repeat):
            new_keywords = self.shuffle_keywords(keywords, hotwords)
            if self.enable_svo and self.enable_keyword:
                encoded_texts.append(self.start_token +
                                     self.build_section('entity_relation', str(svo_triples).strip("[]").replace("'", '')) +
                                     self.build_section('keywords', new_keywords) +
                                     self.build_section('title', title) +
                                     self.build_section('body', body) +
                                     self.end_token + "\n")
            if self.enable_keyword and not self.enable_svo:
                encoded_texts.append(self.start_token +
                                     self.build_section('keywords', new_keywords) +
                                     self.build_section('title', title) +
                                     self.build_section('body', body) +
                                     self.end_token + "\n")
            else:
                # default: svo
                encoded_texts.append(self.start_token +
                                     self.build_section('entity_relation',
                                                        str(svo_triples).strip("[]").replace("'", '')) +
                                     self.build_section('title', title) +
                                     self.build_section('body', body) +
                                     self.end_token + "\n")
        return encoded_texts
