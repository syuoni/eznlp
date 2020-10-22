# -*- coding: utf-8 -*-
from .config import Config

class DecoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CRF')
        self.in_dim = kwargs.pop('in_dim', None)
        self.dropout = kwargs.pop('dropout', 0.5)
        
        self.scheme = kwargs.pop('scheme', 'BIOES')
        self.cascade_mode = kwargs.pop('cascade_mode', 'None')
        
        self.idx2tag = kwargs.pop('idx2tag', None)
        self.tag2idx = kwargs.pop('tag2idx', None)
        self.idx2cas_tag = kwargs.pop('idx2cas_tag', None)
        self.cas_tag2idx = kwargs.pop('cas_tag2idx', None)
        self.idx2cas_type = kwargs.pop('idx2cas_type', None)
        self.cas_type2idx = kwargs.pop('cas_type2idx', None)
        
        super().__init__(**kwargs)
        
        
class TaggerConfig(Config):
    def __init__(self, **kwargs):
        self.embedder = kwargs.pop('embedder', None)
        self.encoders = kwargs.pop('encoders', None)
        self.ptm_encoder = kwargs.pop('ptm_encoder', None)
        self.decoder = kwargs.pop('decoder', None)
        
        super().__init__(**kwargs)
        
    @property
    def is_valid(self):
        is_valid_enc = (self.embedder and self.encoders) or self.ptm_encoder
        is_valid_dec = self.decoder
        return is_valid_enc and is_valid_dec
        
    @property
    def name(self):
        name_elements = []
        if self.embedder is not None and self.embedder.char is not None:
            name_elements.append(self.embedder.char.arch)
        
        if self.encoders is not None:
            name_elements.append(self.encoders.arch)
            
        if self.ptm_encoder is not None:
            name_elements.append(self.ptm_encoder.arch)
            
        name_elements.append(self.decoder.arch)
        return '-'.join(name_elements)
        
    

class ConfigHelper(object):
    @staticmethod
    def load_default_config(config, enc_arches=None, dec_arch=None, ptm=None):
        
        assert (enc_arches is not None) or (ptm is not None)
        # if enc_arches is not None:
            # config['enc'] = [_default_encoder_config(enc_arch) for enc_arch in enc_arches]
        if ptm is not None:
            config['pt_enc'] = {'hid_dim': ptm.config.hidden_size, 
                                'arch': 'PTM'}
            
            
        assert (dec_arch is not None)
        # config['dec'] = _default_decoder_config(dec_arch)
        
        return ConfigHelper.update_dims(config)
    

    @staticmethod
    def update_dims(config):
        if 'emb' in config:
            emb_config = config['emb']
            tok_emb_dim = emb_config['tok']['emb_dim'] if 'tok' in emb_config else 0
            char_out_dim = emb_config['char']['out_dim'] if 'char' in emb_config else 0
            enum_full_dim = sum(enum_config['emb_dim'] for f, enum_config in emb_config['enum'].items()) if 'enum' in emb_config else 0
            val_full_dim = sum(val_config['emb_dim'] for f, val_config in emb_config['val'].items()) if 'val' in emb_config else 0
            emb_dim = tok_emb_dim + char_out_dim + enum_full_dim + val_full_dim
            config['emb']['emb_dim'] = emb_dim
            
            if 'enc' in config:
                for enc_config in config['enc']:
                    enc_config['emb_dim'] = emb_dim
                    if enc_config['arch'].lower() == 'shortcut':
                        enc_config['hid_dim'] = emb_dim
                        
        enc_hid_dim = sum(enc_config['hid_dim'] for enc_config in config['enc']) if 'enc' in config else 0
        pt_enc_hid_dim = config['pt_enc']['hid_dim'] if 'pt_enc' in config else 0
        config['dec']['hid_full_dim'] = enc_hid_dim + pt_enc_hid_dim
        return config
    

    @staticmethod
    def model_name(config, tag_helper):
        name_elements = []
        if 'enc' in config:
            name_elements.extend([enc_config['arch'] for enc_config in config['enc']])
            
        if 'pt_enc' in config:
            name_elements.append(config['pt_enc']['arch'])
            
        name_elements.append(config['dec']['arch'])
        
        if tag_helper.cascade_mode.lower() != 'none':
            name_elements.append(tag_helper.cascade_mode)
        
        return '-'.join(name_elements)
        
        
