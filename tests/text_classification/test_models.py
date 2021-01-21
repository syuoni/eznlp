# -*- coding: utf-8 -*-
import pytest
import torch.optim as optim

from eznlp import EncoderConfig, PreTrainedEmbedderConfig
from eznlp.text_classification import TextClassificationDecoderConfig, TextClassifierConfig
from eznlp.text_classification import TextClassificationDataset
from eznlp.text_classification import TextClassificationTrainer



class TestClassifier(object):
    def one_classifier_pass(self, classifier, dataset, device):
        classifier.eval()
        
        batch012 = dataset.collate([dataset[i] for i in range(0, 3)]).to(device)
        batch123 = dataset.collate([dataset[i] for i in range(1, 4)]).to(device)
        losses012, hidden012 = classifier(batch012, return_hidden=True)
        losses123, hidden123 = classifier(batch123, return_hidden=True)
        
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        assert delta_hidden.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        pred_labels012 = classifier.decode(batch012)
        pred_labels123 = classifier.decode(batch123)
        assert pred_labels012[1:] == pred_labels123[:-1]
        
        optimizer = optim.AdamW(classifier.parameters())
        trainer = TextClassificationTrainer(classifier, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
    
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    @pytest.mark.parametrize("pooling_mode", ['Min', 'Max', 'Mean'])
    def test_classifier(self, yelp2013_demo, enc_arch, pooling_mode, device):
        config = TextClassifierConfig(encoder=EncoderConfig(arch=enc_arch), 
                                      decoder=TextClassificationDecoderConfig(use_attention=False, 
                                                                              pooling_mode=pooling_mode))
        
        dataset = TextClassificationDataset(yelp2013_demo, config)
        classifier = config.instantiate().to(device)
        self.one_classifier_pass(classifier, dataset, device)
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    @pytest.mark.parametrize("attention_scoring", ['Dot', 'Multiplicative', 'Additive'])
    def test_classifier_attention(self, yelp2013_demo, enc_arch, attention_scoring, device):
        config = TextClassifierConfig(encoder=EncoderConfig(arch=enc_arch), 
                                      decoder=TextClassificationDecoderConfig(use_attention=True, 
                                                                              attention_scoring=attention_scoring))
        
        dataset = TextClassificationDataset(yelp2013_demo, config)
        classifier = config.instantiate().to(device)
        self.one_classifier_pass(classifier, dataset, device)
        
    @pytest.mark.parametrize("from_tokenized", [True, False])
    def test_classifier_bert_like(self, yelp2013_demo, bert_with_tokenizer, from_tokenized, device):
        bert, tokenizer = bert_with_tokenizer
        bert_like_embedder_config = PreTrainedEmbedderConfig(arch='BERT', 
                                                             out_dim=bert.config.hidden_size, 
                                                             tokenizer=tokenizer, 
                                                             from_tokenized=from_tokenized)
        config = TextClassifierConfig(encoder=None, bert_like_embedder=bert_like_embedder_config)
        
        dataset = TextClassificationDataset(yelp2013_demo, config)
        classifier = config.instantiate(bert_like=bert).to(device)
        self.one_classifier_pass(classifier, dataset, device)
        
        
        