from typing import Dict
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register('sentiment_plaintext_reader')
class SentimentPlaintextReader(DatasetReader):
    '''
    This is a workaround to be able to make sentiment predictions on
    plaintext sentences against a SST-trained model.

    (To load SST data, e.g., when training or evaluating on the SST, use
    the built-in SST AllenNLP class instead.)

    Reads plaintext input; each line becomes an Instance containing a TextField
    of the line's tokens.

    Usage:
      allennlp predict \
        <model_path> <input_path> \
        --include-package <package_name> \
        --predictor sentiment_predictor \
        --overrides "{dataset_reader: {type: 'sentiment_plaintext_reader'}}"
    '''

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 # We don't actually care about these, but AllenNLP
                 # complains about these extra params being passed -- the
                 # predict command uses the same (SST) config from training to
                 # set this reader up, overrides blob notwithstanding.
                 use_subtrees: bool = False,
                 granularity: str = '2-class') -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = (token_indexers or
                                {'tokens': SingleIdTokenIndexer()})

    @overrides
    def _read(self, path: str):
        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if not line:
                continue

            instance = self.text_to_instance(line)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self, text: str) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        token_field = TextField(tokens, self._token_indexers)

        return Instance({'tokens': token_field})
