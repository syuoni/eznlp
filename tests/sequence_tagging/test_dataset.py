# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.data import Token
from eznlp import ConfigDict, EnumConfig, ValConfig, EmbedderConfig
from eznlp.sequence_tagging import SequenceTaggerConfig, SequenceTaggingDataset


class TestBatch(object):
    @pytest.mark.cuda
    def test_batch_to_cuda(self, conll2003_demo):
        embedder_config = EmbedderConfig(enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token.basic_enum_fields]), 
                                         val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token.basic_val_fields]))
        config = SequenceTaggerConfig(embedder=embedder_config)
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate, pin_memory=True)
        for batch in dataloader:
            break
        
        assert batch.tok_ids.is_pinned()
        assert batch.enum['en_pattern'].is_pinned()
        assert batch.val['en_shape_features'].is_pinned()
        assert batch.seq_lens.is_pinned()
        assert batch.tags_objs[0].tag_ids.is_pinned()
        
        cpu = torch.device('cpu')
        gpu = torch.device('cuda:0')
        assert batch.tok_ids.device == cpu
        batch = batch.to(gpu)
        assert batch.tok_ids.device == gpu
        assert not batch.tok_ids.is_pinned()
        
        
    def test_batches(self, conll2003_demo, device):
        dataset = SequenceTaggingDataset(conll2003_demo)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate)
        for batch in dataloader:
            batch.to(device)
        
        