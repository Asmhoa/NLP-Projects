# Amol Sharma
# Dataset Predictor definition

import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register('twitter_tagger_pred')
class PartsOfSpeechPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return (json.dumps(' '.join(outputs['words'])) + '\t' + json.dumps(' '.join(outputs['tags']))).replace('"', '') + '\n'