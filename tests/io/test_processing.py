# -*- coding: utf-8 -*-
import pytest

from eznlp.io import PostIO


@pytest.mark.parametrize("absorb_attr_types", [[], 
                                               ['Analyzed'], 
                                               ['Analyzed', 'Denied'], 
                                               ['Unconfirmed', 'Analyzed', 'Denied']])
def test_absorb_attributes(absorb_attr_types, HwaMei_demo):
    data = HwaMei_demo
    post_io = PostIO()

    data_abs = post_io.absorb_attributes(data, absorb_attr_types=absorb_attr_types)
    assert all(attr_type not in absorb_attr_types for entry_abs in data_abs for attr_type, ck in entry_abs['attributes'])
    if len(absorb_attr_types) > 0:
        assert len(set(ck[0] for entry in data for ck in entry['chunks'])) < len(set(ck[0] for entry in data_abs for ck in entry['chunks']))
    assert all(ck in entry['chunks'] for entry in data_abs for attr_type, ck in entry['attributes'])
    assert all(head in entry['chunks'] and tail in entry['chunks'] for entry in data_abs for rel_type, head, tail in entry['relations'])

    data_retr = post_io.exclude_attributes(data_abs)
    assert all(entry['tokens'] == entry_retr['tokens'] for entry, entry_retr in zip(data, data_retr))
    assert all(entry['chunks'] == entry_retr['chunks'] for entry, entry_retr in zip(data, data_retr))
    assert all(set(entry['attributes']) == set(entry_retr['attributes']) for entry, entry_retr in zip(data, data_retr))
    assert all(entry['relations'] == entry_retr['relations'] for entry, entry_retr in zip(data, data_retr))


@pytest.mark.parametrize("group_rel_types", [[], 
                                             ['Group_DS', 'Group_Test'], 
                                             ['Group_DS', 'Group_Test', 'Syn_Treat']])
def test_infer_relations(group_rel_types, HwaMei_demo):
    data = HwaMei_demo
    post_io = PostIO()

    data_inf = post_io.infer_relations(data, group_rel_types=group_rel_types)
    assert all(entry['tokens'] == entry_inf['tokens'] for entry, entry_inf in zip(data, data_inf))
    assert all(entry['chunks'] == entry_inf['chunks'] for entry, entry_inf in zip(data, data_inf))
    assert all(entry['attributes'] == entry_inf['attributes'] for entry, entry_inf in zip(data, data_inf))
    assert all(set(entry['relations']).issubset(set(entry_inf['relations'])) for entry, entry_inf in zip(data, data_inf))
    if len(group_rel_types) > 0:
        assert sum(len(entry_inf['relations']) - len(entry['relations']) for entry, entry_inf in zip(data, data_inf)) == 14
