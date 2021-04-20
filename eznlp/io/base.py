# -*- coding: utf-8 -*-

class IO(object):
    """
    An IO interface. 
    
    """
    def __init__(self, encoding=None, verbose: bool=True, **kwargs):
        self.encoding = encoding
        self.verbose = verbose
        self.kwargs = kwargs
        
    def read(self, file_path):
        raise NotImplementedError("Not Implemented `read`")
        