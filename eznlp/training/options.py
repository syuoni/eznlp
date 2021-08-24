# -*- coding: utf-8 -*-
import itertools
import random
import math
import logging

logger = logging.getLogger(__name__)


class OptionSampler(object):
    def __init__(self, **kwargs):
        for arg_name, arg_range in kwargs.items():
            if arg_range is None:
                setattr(self, arg_name, [arg_range])
            elif isinstance(arg_range, (bool, int, float, str)):
                setattr(self, arg_name, [arg_range])
            elif isinstance(arg_range, (list, tuple)):
                assert all(isinstance(arg_value, (bool, int, float, str)) for arg_value in arg_range)
                setattr(self, arg_name, list(arg_range))
            else:
                raise RuntimeError(f"Invalid `arg_range`: {arg_range}")
        
    @property
    def num_possible_options(self):
        return math.prod(len(arg_range) for arg_range in self.__dict__.values())
        
    def _parse_argument(self, arg_name, arg_value):
        if arg_value is None:
            return arg_name
        elif isinstance(arg_value, bool):
            if arg_value:
                return f"--{arg_name}"
            else:
                return ""
        elif isinstance(arg_value, (int, str)):
            return f"--{arg_name} {arg_value}"
        else:
            return f"--{arg_name} {arg_value:.3e}"
        
        
    def _evenly_sample_values(self, arg_range, num):
        num_copies, num_residuals = divmod(num, len(arg_range))
        arg_values = arg_range*num_copies + random.sample(arg_range, num_residuals)
        random.shuffle(arg_values)
        return arg_values
        
        
    def evenly_sample(self, num_options):
        zip_options = [[self._parse_argument(arg_name, arg_value) for arg_value in self._evenly_sample_values(arg_range, num_options)] 
                            for arg_name, arg_range in self.__dict__.items()]
        return list(zip(*zip_options))
        
    def fully_sample(self):
        option_space = [[self._parse_argument(arg_name, arg_value) for arg_value in arg_range] 
                             for arg_name, arg_range in self.__dict__.items()]
        full_options = list(itertools.product(*option_space))
        assert len(full_options) == self.num_possible_options
        return full_options
        
    def randomly_sample(self, num_options):
        # Chooses k unique random elements from a population
        # All sub-slices will also be valid random samples
        return random.sample(self.fully_sample(), num_options)
        
    def sample(self, num_options=None):
        if num_options is None or num_options >= self.num_possible_options:
            logger.info(f"Sampling fully {self.num_possible_options}/{self.num_possible_options} options...")
            return self.fully_sample()
        elif num_options > self.num_possible_options*0.25:
            # `num_options` is much larger than `num_possible_options`, avoid duplicate sampling of an option combination
            logger.info(f"Sampling randomly {num_options}/{self.num_possible_options} options...")
            return self.randomly_sample(num_options)
        else:
            # `num_options` is much smaller than `num_possible_options`, avoid unbalanced sampling for a specific argument
            logger.info(f"Sampling evenly {num_options}/{self.num_possible_options} options...")
            return self.evenly_sample(num_options)



if __name__ == '__main__':
    import numpy
    logging.basicConfig(level=logging.INFO)
    sampler = OptionSampler(num_epochs=[50, 100], 
                            batch_size=32,
                            lr=numpy.logspace(-3, -5, num=9, base=10).tolist(), 
                            arch=['LSTM', 'CNN'],
                            fs=None, 
                            no_bert=[True, False])
    for option in sampler.sample(100):
        print(option)
