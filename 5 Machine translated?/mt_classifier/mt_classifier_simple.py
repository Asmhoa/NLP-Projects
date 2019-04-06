from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("mt_classifier_3")
class MTClassifier3(Model):
    """
    This ``Model`` performs text classification for source and candidate translation pairs.  We assume we're given a
    source and an candidate, and we predict some output label.
    The basic model structure: we'll embed the source and the candidate, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    source_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    candidate_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the abstract to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 source_encoder: Seq2VecEncoder,
                 candidate_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MTClassifier3, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.source_encoder = source_encoder
        self.candidate_encoder = candidate_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != source_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            source_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != candidate_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            candidate_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                candidate: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        source : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        candidate : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        metadata : List[Dict[str, Any]], optionsl (default = None)
            Metadata for use by predictors
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_source = self.text_field_embedder(source)
        source_mask = util.get_text_field_mask(source)
        encoded_source = self.source_encoder(embedded_source, source_mask)

        embedded_candidate = self.text_field_embedder(candidate)
        candidate_mask = util.get_text_field_mask(candidate)
        encoded_candidate = self.candidate_encoder(embedded_candidate, candidate_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_source, encoded_candidate], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        if metadata is not None:
            output_dict["source"] = [x["source"] for x in metadata]
            output_dict["candidate"] = [x["candidate"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}