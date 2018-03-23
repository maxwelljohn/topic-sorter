#!/usr/bin/env python

import nltk
import numpy as np
import re

PASSAGE_SEPARATOR = "\n\n"
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS.add('http')
STOPWORDS.add('https')
WORD_RE = re.compile('^\w+$')
MAX_NGRAM_N = 3


def main(passage_file):
        passages = passage_file.read().split(PASSAGE_SEPARATOR)
        wnl = nltk.WordNetLemmatizer()
        passage_ngrams = {}
        ngram_document_frequency = nltk.FreqDist()

        for passage in passages:
            lemmas = [wnl.lemmatize(t) for t in
                      nltk.word_tokenize(passage.lower())]
            lemmas = [l for l in lemmas if l not in STOPWORDS and
                      re.match(WORD_RE, l)]

            ngrams = []
            for n in range(1, MAX_NGRAM_N+1):
                ngrams.extend(nltk.ngrams(lemmas, n))
            passage_ngrams[passage] = nltk.FreqDist(ngrams)

            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                ngram_document_frequency[ngram] += 1

        similarity = {}
        for index1, passage1 in enumerate(passages):
            for passage2 in passages[index1+1:]:
                canonical_order = tuple(sorted([passage1, passage2]))
                ngrams1 = passage_ngrams[passage1]
                ngrams2 = passage_ngrams[passage2]
                similarity_score = 0
                for g in ngrams1:
                    if ngrams2[g] > 0:
                        # TF-IDF weighting
                        similarity_score += (1 + np.log(ngrams1[g])) * \
                            (1 + np.log(ngrams2[g])) * \
                            (np.log(len(passages)/ngram_document_frequency[g]))
                similarity[canonical_order] = similarity_score

    pairs_w_similarity = list(similarity.items())
    pairs_w_similarity.sort(key=lambda ps: ps[1])
    print("### MOST SIMILAR ###")
    displayed = 0
    for p, s in reversed(pairs_w_similarity):
        # Skip long passages to make evaluation easier.
        if displayed < 10 and (len(p[0]) + len(p[1])) < 1000:
                print(p[0])
                print('---')
                print(p[1])
                print('')
                print('')
                displayed += 1
    displayed = 0
    print("### LEAST SIMILAR ###")
    for p, s in pairs_w_similarity:
        if displayed < 10 and (len(p[0]) + len(p[1])) < 1000:
                print(p[0])
                print('---')
                print(p[1])
                print('')
                print('')
                displayed += 1


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '-':
        passage_file = sys.stdin
        main(passage_file)
    else:
        passage_filepath = sys.argv[1]
        with open(passage_filepath, 'r') as passage_file:
            main(passage_file)
