from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('sentiment_predictor')
class SentimentPredictor(Predictor):
    '''
    Basic predictor class for use with the SentimentPlaintextReader.

    Usage:
      allennlp predict \
        <model_path> <input_path> \
        --include-package <package_name> \
        --predictor sentiment_predictor \
        --overrides "{dataset_reader: {type: 'sentiment_plaintext_reader'}}"
    '''

    def predict(self, text: str) -> JsonDict:
        return self.predict_json({'text': text})

    @overrides
    def load_line(self, line: str) -> JsonDict:
        # Since we don't have any input fields besides the sentence itself,
        # it doesn't really make sense to pack things in json -- just have each
        # input line be a sentence we want to classify.
        return {'text': line}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict['text'])
        return instance
