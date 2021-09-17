# -*- coding: utf-8 -*-
import pytest
import numpy
import nltk
import torchtext

from eznlp.metrics import precision_recall_f1_report
from eznlp.utils import ChunksTagsTranslator
from eznlp.io import ConllIO


class TestMetric(object):
    def _assert_scores_equal(self, ave_scores, expected_ave_scores):
        for key in ave_scores:
            for score_key in ['precision', 'recall', 'f1']:
                assert numpy.abs(ave_scores[key][score_key] - expected_ave_scores[key][score_key]) < 1e-6
        
        
    @pytest.mark.parametrize("tags_gold_data, tags_pred_data, expected_ave_scores", 
                             [([['B-A', 'B-B', 'O', 'B-A']], 
                               [['O', 'B-B', 'B-C', 'B-A']], 
                               {'macro': {'precision': 2/3, 'recall': 1/2, 'f1': 5/9}, 
                                'micro': {'precision': 2/3, 'recall': 2/3, 'f1': 2/3}}), 
                              ([['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], 
                                ['B-PER', 'I-PER', 'O']], 
                               [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], 
                                ['B-PER', 'I-PER', 'O']], 
                               {'macro': {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}, 
                                'micro': {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}})])
    def test_example(self, tags_gold_data, tags_pred_data, expected_ave_scores):
        translator = ChunksTagsTranslator(scheme='BIO2')
        chunks_gold_data = [translator.tags2chunks(tags) for tags in tags_gold_data]
        chunks_pred_data = [translator.tags2chunks(tags) for tags in tags_pred_data]
        
        scores, ave_scores = precision_recall_f1_report(chunks_gold_data, chunks_pred_data)
        self._assert_scores_equal(ave_scores, expected_ave_scores)
        
        
    def test_conll2000(self):
        # https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        # https://github.com/chakki-works/seqeval
        gold_data = ConllIO(text_col_id=0, tag_col_id=2, scheme='BIO2').read("data/conlleval/output.txt")
        pred_data = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO2').read("data/conlleval/output.txt")
        
        chunks_gold_data = [ex['chunks'] for ex in gold_data]
        chunks_pred_data = [ex['chunks'] for ex in pred_data]
        
        expected_ave_scores = {'macro': {'precision': 0.54880502,
                                         'recall': 0.58776420, 
                                         'f1': 0.55397552}, 
                               'micro': {'precision': 0.68831169, 
                                         'recall': 0.80827887, 
                                         'f1': 0.74348697}}
        scores, ave_scores = precision_recall_f1_report(chunks_gold_data, chunks_pred_data)
        self._assert_scores_equal(ave_scores, expected_ave_scores)



def test_bleu_score():
    # `nltk` and `torchtext` results are consistent if all candidate sentences are longer than 4. 
    # `torchtext.data.bleu_score` raise errors if the input sentences include spaces
    candidate_corpus = [['my', 'full', 'pytorch', 'test'], 
                        ['this', 'is', 'another', 'sentence']]
    references_corpus = [[['my', 'full', 'pytorch', 'test'], ['completely', 'different']], 
                         [['this', 'is']]]
    torchtext_bleu = torchtext.data.bleu_score(candidate_corpus, references_corpus)
    nltk_bleu = nltk.translate.bleu_score.corpus_bleu(references_corpus, candidate_corpus)
    
    assert abs(torchtext_bleu - nltk_bleu) < 1e-6
