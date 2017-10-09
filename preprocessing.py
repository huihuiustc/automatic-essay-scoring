import re
import nltk
import copy
from nltk import ngrams
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import Term


def data_set_preprocessing(essay_list, stopword):
    for essay in essay_list:
        essay_case_folded = case_folding(essay.essay_content)
        term_list_tokenize = tokenizing(essay_case_folded)
        term_list_removed_stopword = stopword_removal(stopword, term_list_tokenize)
        term_list_stemming = stemming(term_list_removed_stopword)
        term_list_ngram = ngram(term_list_stemming)
        term_list = normalize_term_list(term_list_ngram)
        essay.set_term_list(term_list)


def case_folding(essay):
    return re.sub('[^a-z]+', ' ', essay.lower())


def tokenizing(essay):
    term_list_tokenize = []
    for item in nltk.word_tokenize(essay):
        term = Term.Term(item)
        term_list_tokenize.append(term)
    return term_list_tokenize


def stopword_removal(stopword, term_list):
    term_list_removed_stopword = []
    for term in term_list:
        if term.term_content not in stopword:
            term_list_removed_stopword.append(term)
    return term_list_removed_stopword


def stemming(term_list):
    term_list_stemming = []
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    for term in term_list:
        output = stemmer.stem(str(term.term_content))
        term_list_stemming.append(Term.Term(output))
    return term_list_stemming


def ngram(term_list):
    term_list_ngram = copy.deepcopy(term_list)
    sentence = ''
    for term in term_list:
        sentence += str(term.term_content)+' '

    bigram = ngrams(sentence.split(), 2)
    for gram in bigram:
        string = re.sub('[^a-z ]+', '', str(gram))
        term_list_ngram.append(Term.Term(string))
    return term_list_ngram


def normalize_term_list(term_list):
    existed_term = []
    term_content_in_string_array = convert_term_content_to_string_array(term_list)
    new_term_list = []
    max_tf = 0

    for term in term_list:
        if term.term_content not in existed_term:
            existed_term.append(term.term_content)
            t = Term.Term(term.term_content)
            t.set_tf(term_content_in_string_array.count(t.term_content))
            new_term_list.append(t)
            if t.tf > max_tf:
                max_tf = t.tf

    for term in new_term_list:
        term.set_ntf(float(term.tf) / float(max_tf))

    return new_term_list


def convert_term_content_to_string_array(term_list):
    term_content_in_string_array = []
    for term in term_list:
        term_content_in_string_array.append(term.term_content)
    return term_content_in_string_array
