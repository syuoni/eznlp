# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.config import ConfigDict
from eznlp.dataset import Dataset
from eznlp.model import ExtractorConfig, MultiHotConfig, OneHotConfig
from eznlp.token import Token


def test_batch_to_cuda(conll2003_demo, device):
    if device.type.startswith("cpu"):
        pytest.skip("test requires cuda, while current session runs on cpu")

    torch.cuda.set_device(device)
    config = ExtractorConfig(
        "sequence_tagging",
        ohots=ConfigDict(
            {f: OneHotConfig(field=f, emb_dim=20) for f in Token._basic_ohot_fields}
        ),
        mhots=ConfigDict(
            {f: MultiHotConfig(field=f, emb_dim=20) for f in Token._basic_mhot_fields}
        ),
    )
    dataset = Dataset(conll2003_demo, config)
    dataset.build_vocabs_and_dims()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate, pin_memory=True
    )
    for batch in dataloader:
        break

    assert batch.ohots["text"].is_pinned()
    assert batch.ohots["en_pattern"].is_pinned()
    assert batch.mhots["en_shape_features"].is_pinned()
    assert batch.seq_lens.is_pinned()
    assert batch.tags_objs[0].tag_ids.is_pinned()

    assert batch.ohots["text"].device.type.startswith("cpu")
    batch = batch.to(device)
    assert batch.ohots["text"].device.type.startswith("cuda")
    assert not batch.ohots["text"].is_pinned()


def test_batches(conll2003_demo, device):
    dataset = Dataset(conll2003_demo, ExtractorConfig("sequence_tagging"))
    dataset.build_vocabs_and_dims()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate
    )
    for batch in dataloader:
        batch.to(device)
