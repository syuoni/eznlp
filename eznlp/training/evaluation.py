# -*- coding: utf-8 -*-
import logging

from ..metrics import precision_recall_f1_report
from ..dataset import Dataset
from .trainer import Trainer


logger = logging.getLogger(__name__)


def evaluate_text_classification(trainer: Trainer, dataset: Dataset):
    set_labels_pred = trainer.predict(dataset)
    set_labels_gold = [ex['label'] for ex in dataset.data]
    
    acc = trainer.model.decoder.evaluate(set_labels_gold, set_labels_pred)
    logger.info(f"TC Accuracy: {acc*100:2.3f}%")


def evaluate_entity_recognition(trainer: Trainer, dataset: Dataset):
    set_chunks_pred = trainer.predict(dataset)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"ER Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"ER Macro F1-score: {macro_f1*100:2.3f}%")


def evaluate_relation_extraction(trainer: Trainer, dataset: Dataset):
    set_relations_pred = trainer.predict(dataset)
    set_relations_gold = [ex['relations'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_relations_gold, set_relations_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"RE Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"RE Macro F1-score: {macro_f1*100:2.3f}%")


def evaluate_joint_er_re(trainer: Trainer, dataset: Dataset):
    set_chunks_pred, set_relations_pred = trainer.predict(dataset)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]
    set_relations_gold = [ex['relations'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"ER Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"ER Macro F1-score: {macro_f1*100:2.3f}%")
    
    scores, ave_scores = precision_recall_f1_report(set_relations_gold, set_relations_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"RE Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"RE Macro F1-score: {macro_f1*100:2.3f}%")
