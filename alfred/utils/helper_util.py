import torch

from vocab import Vocab as VocabBase
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.6f} seconds')
        return result
    return timeit_wrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class VocabWithLock(VocabBase):
    ''' vocab.Vocab with a lock for parallel computations. '''
    def __init__(self, words=(), lock=None):
        # super(VocabBase, self).__init__(words)
        self.lock = lock
        super().__init__(words)

    def word2index(self, word, train=False):
        ''' Original function copy with the self.lock call. '''
        if isinstance(word, (list, tuple)):
            return [self.word2index(w, train=train) for w in word]
        with self.lock:
            self.counts[word] += train
            if word in self._word2index:
                return self._word2index[word]
            else:
                if train:
                    self._index2word += [word]
                    self._word2index[word] = len(self._word2index)
                else:
                    return self._handle_oov_word(word)
            index = self._word2index[word]
        return index


def identity(x):
    '''
    pickable equivalent of lambda x: x
    '''
    return x
