# Amol Sharma
# Dataset Reader definition

from typing import List, Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides


@DatasetReader.register("twitter_tagger_read")
class PartsOfSpeechDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        sentence, tags = [], []
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file:
                tokens = line.strip().split()
                if len(tokens) == 0:
                    yield self.text_to_instance(sentence, tags)
                    sentence, tags = [], []
                elif tokens[0].isdigit():
                    sentence.append(tokens[1])
                    tags.append(tokens[3])

    @overrides
    def text_to_instance(self, words: List[str], tags: List[str]) -> Instance:
        # Convert each word to token
        tokenized_sentence = [Token(w) for w in words]
        sentence_text_field = TextField(tokenized_sentence, self._token_indexers)
        fields = {'tokens': sentence_text_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_text_field)
            fields['tags'] = label_field

        fields['metadata'] = MetadataField({'words': words})
        return Instance(fields)
