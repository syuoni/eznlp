# -*- coding: utf-8 -*-
import logging
import nltk

from ..metrics import precision_recall_f1_report
from ..dataset import Dataset
from .trainer import Trainer


logger = logging.getLogger(__name__)


def evaluate_text_classification(trainer: Trainer, dataset: Dataset, batch_size: int=32):
    set_labels_pred = trainer.predict(dataset, batch_size=batch_size)
    set_labels_gold = [ex['label'] for ex in dataset.data]
    
    acc = trainer.model.decoder.evaluate(set_labels_gold, set_labels_pred)
    logger.info(f"TC | Accuracy: {acc*100:2.3f}%")


def disp_prf(ave_scores: dict, task: str='ER'):
    for key_text, key in zip(['Precision', 'Recall', 'F1-score'], ['precision', 'recall', 'f1']):
        logger.info(f"{task} | Micro {key_text}: {ave_scores['micro'][key]*100:2.3f}%")
    for key_text, key in zip(['Precision', 'Recall', 'F1-score'], ['precision', 'recall', 'f1']):
        logger.info(f"{task} | Macro {key_text}: {ave_scores['macro'][key]*100:2.3f}%")


def evaluate_entity_recognition(trainer: Trainer, dataset: Dataset, batch_size: int=32):
    set_chunks_pred = trainer.predict(dataset, batch_size=batch_size)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
    disp_prf(ave_scores, task='ER')


def evaluate_attribute_extraction(trainer: Trainer, dataset: Dataset, batch_size: int=32):
    set_attributes_pred = trainer.predict(dataset, batch_size=batch_size)
    set_attributes_gold = [ex['attributes'] for ex in dataset.data]

    scores, ave_scores = precision_recall_f1_report(set_attributes_gold, set_attributes_pred)
    disp_prf(ave_scores, task='AE')


def evaluate_relation_extraction(trainer: Trainer, dataset: Dataset, eval_chunk_type_for_relation: bool=True, batch_size: int=32):
    set_relations_pred = trainer.predict(dataset, batch_size=batch_size)
    set_relations_gold = [ex['relations'] for ex in dataset.data]
    
    if not eval_chunk_type_for_relation:
        set_relations_gold = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_relations_gold]
        set_relations_pred = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_relations_pred]
    
    scores, ave_scores = precision_recall_f1_report(set_relations_gold, set_relations_pred)
    disp_prf(ave_scores, task='RE')


def evaluate_joint_extraction(trainer: Trainer, dataset: Dataset, has_attr: bool=False, has_rel: bool=True, eval_chunk_type_for_relation: bool=True, batch_size: int=32):
    sets_pred = trainer.predict(dataset, batch_size=batch_size)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]

    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, sets_pred[0])
    disp_prf(ave_scores, task='ER')

    if has_attr:
        set_attributes_gold = [ex['attributes'] for ex in dataset.data]
        scores, ave_scores = precision_recall_f1_report(set_attributes_gold, sets_pred[1])
        disp_prf(ave_scores, task='AE')

    if has_rel:
        set_relations_gold = [ex['relations'] for ex in dataset.data]
        set_relations_pred = sets_pred[2] if has_attr else sets_pred[1]

        if not eval_chunk_type_for_relation:
            set_relations_gold = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_relations_gold]
            set_relations_pred = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_relations_pred]

        scores, ave_scores = precision_recall_f1_report(set_relations_gold, set_relations_pred)
        disp_prf(ave_scores, task='RE')



def evaluate_generation(trainer: Trainer, dataset: Dataset, batch_size: int=32, beam_size: int=1):
    set_trg_pred = trainer.predict(dataset, batch_size=batch_size, beam_size=beam_size)
    set_trg_gold = [[tokens.text for tokens in ex['full_trg_tokens']] for ex in dataset.data]
    
    bleu4 = nltk.translate.bleu_score.corpus_bleu(list_of_references=set_trg_gold, hypotheses=set_trg_pred)
    logger.info(f"Beam Size: {beam_size} | BLEU-4: {bleu4*100:2.3f}%")
