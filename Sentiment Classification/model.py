# Edited from https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt1.md

from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("sentiment_classifier")
class SentimentClassifier(Model):
    def __init__(self,
                vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                sentence_encoder: Seq2VecEncoder, # specify this as RNN
                classifier_feedforward: FeedForward) -> None:
        super(SentimentClassifier, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

        self.loss = torch.nn.CrossEntropyLoss()

    # @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_sentence = self.text_field_embedder(tokens)
        sentence_mask = util.get_text_field_mask(tokens)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)

        # TODO: ask about DIMS and what is input_size in encoder for config file
        # original: logits = self.classifier_feedforward(encoded_sentence, dim=-1)
        logits = self.classifier_feedforward(encoded_sentence)

        # original: class_probabilities = F.softmax(logits, dim = -1)
        class_probabilities = F.softmax(logits, dim = -1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    # @overrides 
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict