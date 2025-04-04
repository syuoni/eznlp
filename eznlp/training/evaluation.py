# -*- coding: utf-8 -*-
import logging

import nltk

from ..dataset import Dataset
from ..metrics import precision_recall_f1_report
from ..utils.chunk import detect_nested
from .trainer import Trainer

logger = logging.getLogger(__name__)


def evaluate_text_classification(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32, save_preds: bool = False
):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, label_pred in zip(dataset.data, set_y_pred):
            ex["label_pred"] = label_pred
        logger.info("TC | Predictions saved")
    else:
        set_y_gold = [ex["label"] for ex in dataset.data]
        acc = trainer.model.decoder.evaluate(set_y_gold, set_y_pred)
        logger.info(f"TC | Accuracy: {acc*100:2.3f}%")


def _disp_prf(ave_scores: dict, task: str = "ER"):
    for key_text, key in zip(
        ["Precision", "Recall", "F1-score"], ["precision", "recall", "f1"]
    ):
        logger.info(f"{task} | Micro {key_text}: {ave_scores['micro'][key]*100:2.3f}%")
    for key_text, key in zip(
        ["Precision", "Recall", "F1-score"], ["precision", "recall", "f1"]
    ):
        logger.info(f"{task} | Macro {key_text}: {ave_scores['macro'][key]*100:2.3f}%")


def _eval_ent(set_y_gold, set_y_pred, eval_inex: bool = False):
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task="ER")

    if eval_inex:
        set_y_pred_in = [
            detect_nested(y_pred, y_gold)
            for y_gold, y_pred in zip(set_y_gold, set_y_pred)
        ]
        set_y_gold_in = [detect_nested(y_gold, y_gold) for y_gold in set_y_gold]
        scores, ave_scores = precision_recall_f1_report(set_y_gold_in, set_y_pred_in)
        _disp_prf(ave_scores, task="ER-in")

        set_y_pred_ex = [
            list(set(y_pred) - set(y_pred_in))
            for y_pred, y_pred_in in zip(set_y_pred, set_y_pred_in)
        ]
        set_y_gold_ex = [
            list(set(y_gold) - set(y_gold_in))
            for y_gold, y_gold_in in zip(set_y_gold, set_y_gold_in)
        ]
        scores, ave_scores = precision_recall_f1_report(set_y_gold_ex, set_y_pred_ex)
        _disp_prf(ave_scores, task="ER-ex")


def evaluate_entity_recognition(
    trainer: Trainer,
    dataset: Dataset,
    batch_size: int = 32,
    eval_inex: bool = False,
    pp_callback=None,
    save_preds: bool = False,
):
    """Evaluation of entity recognition results.

    Parameters
    ----------
    eval_inex: bool
        Evaluate internal/external-entity results.
    pp_callback: None or Callable
        Post-processing function applied to predicted results.
    save_preds: bool
        Save the predicted results into `dataset.data`; it is typically used when ground truth is not available offline.
    """
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, chunks_pred in zip(dataset.data, set_y_pred):
            ex["chunks_pred"] = chunks_pred
        logger.info("ER | Predictions saved")
    else:
        set_y_gold = [ex["chunks"] for ex in dataset.data]
        _eval_ent(set_y_gold, set_y_pred, eval_inex=eval_inex)
        if callable(pp_callback):
            logger.info("Post-processing predictions...")
            set_y_pred = [pp_callback(y_pred) for y_pred in set_y_pred]
            _eval_ent(set_y_gold, set_y_pred, eval_inex=eval_inex)


def _eval_attr(set_y_gold, set_y_pred):
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task="AE+")

    set_y_gold = [
        [(attr_type, chunk[1:]) for attr_type, chunk in attributes]
        for attributes in set_y_gold
    ]
    set_y_pred = [
        [(attr_type, chunk[1:]) for attr_type, chunk in attributes]
        for attributes in set_y_pred
    ]
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task="AE")


def _eval_rel(set_y_gold, set_y_pred):
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task="RE+")

    set_y_gold = [
        [(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations]
        for relations in set_y_gold
    ]
    set_y_pred = [
        [(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations]
        for relations in set_y_pred
    ]
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task="RE")


def evaluate_attribute_extraction(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32, save_preds: bool = False
):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, attrs_pred in zip(dataset.data, set_y_pred):
            ex["attributes_pred"] = attrs_pred
        logger.info("AE | Predictions saved")
    else:
        set_y_gold = [ex["attributes"] for ex in dataset.data]
        _eval_attr(set_y_gold, set_y_pred)


def evaluate_relation_extraction(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32, save_preds: bool = False
):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, rels_pred in zip(dataset.data, set_y_pred):
            ex["relations_pred"] = rels_pred
        logger.info("RE | Predictions saved")
    else:
        set_y_gold = [ex["relations"] for ex in dataset.data]
        _eval_rel(set_y_gold, set_y_pred)


def evaluate_joint_extraction(
    trainer: Trainer,
    dataset: Dataset,
    has_attr: bool = False,
    has_rel: bool = True,
    batch_size: int = 32,
    save_preds: bool = False,
):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    set_chunks_pred = set_y_pred[0]
    if has_attr:
        set_attrs_pred = set_y_pred[1]
    if has_rel:
        set_rels_pred = set_y_pred[2] if has_attr else set_y_pred[1]

    if save_preds:
        for ex, chunks_pred in zip(dataset.data, set_chunks_pred):
            ex["chunks_pred"] = chunks_pred
        if has_attr:
            for ex, attrs_pred in zip(dataset.data, set_attrs_pred):
                ex["attributes_pred"] = attrs_pred
        if has_rel:
            for ex, rels_pred in zip(dataset.data, set_rels_pred):
                ex["relations_pred"] = rels_pred
        logger.info("Joint | Predictions saved")
    else:
        set_chunks_gold = [ex["chunks"] for ex in dataset.data]
        _eval_ent(set_chunks_gold, set_chunks_pred, eval_inex=False)
        if has_attr:
            set_attrs_gold = [ex["attributes"] for ex in dataset.data]
            _eval_attr(set_attrs_gold, set_attrs_pred)
        if has_rel:
            set_rels_gold = [ex["relations"] for ex in dataset.data]
            _eval_rel(set_rels_gold, set_rels_pred)


def evaluate_generation(
    trainer: Trainer, dataset: Dataset, batch_size: int = 32, beam_size: int = 1
):
    set_trg_pred = trainer.predict(dataset, batch_size=batch_size, beam_size=beam_size)
    set_trg_gold = [
        [tokens.text for tokens in ex["full_trg_tokens"]] for ex in dataset.data
    ]

    bleu4 = nltk.translate.bleu_score.corpus_bleu(
        list_of_references=set_trg_gold, hypotheses=set_trg_pred
    )
    logger.info(f"Beam Size: {beam_size} | BLEU-4: {bleu4*100:2.3f}%")
