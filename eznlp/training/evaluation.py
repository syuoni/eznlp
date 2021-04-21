# -*- coding: utf-8 -*-
from typing import List
import logging

from ..metrics import precision_recall_f1_report
from ..data.dataset import Dataset
from .trainer import Trainer


logger = logging.getLogger(__name__)


def evaluate_text_classification(trainier: Trainer, dataset: Dataset):
    set_labels_pred = trainier.predict_labels(dataset)
    set_labels_gold = [ex['label'] for ex in dataset.data]
    
    acc = trainier.evaluate(set_labels_gold, set_labels_pred)
    logger.info(f"Accuracy: {acc*100:2.3f}%")


def evaluate_entity_recognition(trainer: Trainer, dataset: Dataset):
    set_chunks_pred = trainer.predict_chunks(dataset)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"Macro F1-score: {macro_f1*100:2.3f}%")


def evaluate_relation_extraction(trainer: Trainer, dataset: Dataset):
    set_relations_pred = trainer.predict_relations(dataset)
    set_relations_gold = [ex['relations'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_relations_gold, set_relations_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"Macro F1-score: {macro_f1*100:2.3f}%")


def union_set_chunks(set_chunks_gold: List[tuple], set_chunks_pred: List[tuple]):
    set_chunks_union = []
    for chunks_gold, chunks_pred in zip(set_chunks_gold, set_chunks_pred):
        existing_spans = {(start, end) for ck_type, start, end in chunks_gold}
        chunks_union = chunks_gold + [(ck_type, start, end) for ck_type, start, end in chunks_pred if (start, end) not in existing_spans]
        set_chunks_union.append(chunks_union)
    return set_chunks_union


