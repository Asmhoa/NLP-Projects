from typing import Dict, Optional, Union, List, Any

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F
# from torchnlp.word_to_vector import FastText

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("mt_classifier_model")
class MTClassifier2(Model):
    """
    This class implements the Biattentive Classification Network model described
    in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    <https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
    given a piece of text, and we predict some output label.

    At a high level, the model starts by embedding the tokens and running them through
    a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
    representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
    on the encoder output representations (self-attention in this case, since
    the two representations that typically go into biattention are identical) and
    get out an attentive vector representation of the text. We combine this text
    representation with the encoder outputs computed earlier, and then run this through
    yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
    integrator and max, min, mean, and self-attention pool to create a final representation,
    which is passed through a maxout network or some feed-forward layers
    to output a classification (``output_layer``).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    source_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``source`` ``TextField`` we get as input to the model.
    candidate_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``candidate`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward_source : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder_source : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    pre_encode_feedforward_candidate : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder_candidate : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator_source : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the source encodings.
    integrator_candidate : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the candidate encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    elmo : ``Elmo``, optional (default=``None``)
        If provided, will be used to concatenate pretrained ELMo representations to
        either the integrator output (``use_integrator_output_elmo``) or the
        input (``use_input_elmo``).
    use_input_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the input vectors.
    use_integrator_output_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the integrator output.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_field_embedder: TextFieldEmbedder,
                 candidate_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 pre_encode_feedforward_source: FeedForward,
                 encoder_source: Seq2SeqEncoder,
                 pre_encode_feedforward_candidate: FeedForward,
                 encoder_candidate: Seq2SeqEncoder,
                 integrator_source: Seq2SeqEncoder,
                 integrator_candidate: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: Union[FeedForward, Maxout],
                 elmo: Elmo,
                 use_input_elmo: bool = False,
                 use_integrator_output_elmo: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MTClassifier2, self).__init__(vocab, regularizer)

        self._source_field_embedder = source_field_embedder
        self._candidate_field_embedder = candidate_field_embedder
        if "elmo" in self._source_field_embedder._token_embedders.keys():  # pylint: disable=protected-access
            raise ConfigurationError("To use ELMo in the BiattentiveClassificationNetwork input, "
                                     "remove elmo from the text_field_embedder and pass an "
                                     "Elmo object to the BiattentiveClassificationNetwork and set the "
                                     "'use_input_elmo' and 'use_integrator_output_elmo' flags accordingly.")

        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward_source = pre_encode_feedforward_source
        self._pre_encode_feedforward_candidate = pre_encode_feedforward_candidate

        self._encoder_source = encoder_source
        self._encoder_candidate = encoder_candidate

        self._integrator_source = integrator_source
        self._integrator_candidate = integrator_candidate
        self._integrator_dropout = nn.Dropout(integrator_dropout)

        self._elmo = elmo
        self._use_input_elmo = use_input_elmo
        self._use_integrator_output_elmo = use_integrator_output_elmo
        self._num_elmo_layers = int(self._use_input_elmo) + int(self._use_integrator_output_elmo)
        
        # Check that, if elmo is None, none of the elmo flags are set.
        if self._elmo is None and self._num_elmo_layers != 0:
            raise ConfigurationError("One of 'use_input_elmo' or 'use_integrator_output_elmo' is True, "
                                     "but no Elmo object was provided upon construction. Pass in an Elmo "
                                     "object to use Elmo.")

        if self._elmo is not None:
            # Check that, if elmo is not None, we use it somewhere.
            if self._num_elmo_layers == 0:
                raise ConfigurationError("Elmo object provided upon construction, but both 'use_input_elmo' "
                                         "and 'use_integrator_output_elmo' are 'False'. Set one of them to "
                                         "'True' to use Elmo, or do not provide an Elmo object upon construction.")
            # Check that the number of flags set is equal to the num_output_representations of the Elmo object
            # pylint: disable=protected-access,too-many-format-args
            if len(self._elmo._scalar_mixes) != self._num_elmo_layers:
                raise ConfigurationError("Elmo object has num_output_representations=%s, but this does not "
                                         "match the number of use_*_elmo flags set to true. use_input_elmo "
                                         "is %s, and use_integrator_output_elmo is %s".format(
                                                 str(len(self._elmo._scalar_mixes)),
                                                 str(self._use_input_elmo),
                                                 str(self._use_integrator_output_elmo)))

        self._combined_integrator_output_dim = self._integrator_source.get_output_dim()

        self._self_attentive_pooling_projection_source = nn.Linear(
                self._combined_integrator_output_dim, 1)
        self._self_attentive_pooling_projection_candidate = nn.Linear(
            self._combined_integrator_output_dim, 1)
        self._output_layer = output_layer

        if self._use_input_elmo:
            check_dimensions_match(candidate_field_embedder.get_output_dim() +
                                   self._elmo.get_output_dim(),
                                   self._pre_encode_feedforward_candidate.get_input_dim(),
                                   "text field embedder output dim + ELMo output dim",
                                   "Pre-encoder feedforward input dim")
        else:
            check_dimensions_match(candidate_field_embedder.get_output_dim(),
                                   self._pre_encode_feedforward_candidate.get_input_dim(),
                                   "text field embedder output dim",
                                   "Pre-encoder feedforward input dim")

        check_dimensions_match(self._pre_encode_feedforward_candidate.get_output_dim(),
                               self._encoder_candidate.get_input_dim(),
                               "Pre-encoder feedforward output dim",
                               "Encoder input dim")
        check_dimensions_match(self._encoder_candidate.get_output_dim() * 3,
                               self._integrator_candidate.get_input_dim(),
                               "Encoder output dim * 3",
                               "Integrator input dim")
        if self._use_integrator_output_elmo:
            check_dimensions_match(self._combined_integrator_output_dim * 4,
                                   self._output_layer.get_input_dim(),
                                   "(Integrator output dim + ELMo output dim) * 4",
                                   "Output layer input dim")
        else:
            check_dimensions_match(self._integrator_candidate.get_output_dim() * 4,
                                   self._output_layer.get_input_dim(),
                                   "Integrator output dim * 4",
                                   "Output layer input dim")

        check_dimensions_match(self._output_layer.get_output_dim(),
                               self._num_classes,
                               "Output layer output dim",
                               "Number of classes.")

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
        source : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        candidate : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        source_mask = util.get_text_field_mask(source).float()
        candidate_mask = util.get_text_field_mask(candidate).float()
        
        # Pop elmo tokens, since elmo embedder should not be present.
        elmo_tokens_candidate = candidate.pop("elmo", None)

        embedded_source = self._source_field_embedder(source)
        embedded_candidate = self._candidate_field_embedder(candidate)

        dropped_embedded_source = self._embedding_dropout(embedded_source)
        dropped_embedded_candidate = self._embedding_dropout(embedded_candidate)
        pre_encoded_source = self._pre_encode_feedforward_source(dropped_embedded_source)
        encoded_source = self._encoder_source(pre_encoded_source, source_mask)
        pre_encoded_candidate = self._pre_encode_feedforward_candidate(dropped_embedded_candidate)
        encoded_candidate = self._encoder_candidate(pre_encoded_candidate, candidate_mask)

        # Compute biattention. This is a special case since the inputs are the same.
 
        attention_logits_source = encoded_source.bmm(encoded_candidate.permute(0, 2, 1).contiguous())
        attention_logits_candidate = attention_logits_source.permute(0, 2, 1).contiguous()
        # print("Attention Logits for Source Size: ", attention_logits_source.size())
        # print("Attention Logits for Candidate Size: ", attention_logits_candidate.size())
        # print("Source Mask Size: ", source_mask.size())
        # print("Candidate Mask Size: ", candidate_mask.size())
        attention_weights_source = util.masked_softmax(attention_logits_source.permute(0, 2, 1), source_mask)
        attention_weights_candidate = util.masked_softmax(attention_logits_candidate.permute(0, 2, 1), candidate_mask)
        encoded_source = util.weighted_sum(encoded_source, attention_weights_source)
        encoded_candidate = util.weighted_sum(encoded_candidate, attention_weights_candidate)

        max_second_dim = max(encoded_candidate.size()[1], encoded_source.size()[1])
        # print("\nMAX", max_second_dim)
        encoded_candidate = torch.cat([encoded_candidate,
            torch.cuda.FloatTensor(encoded_candidate.size()[0], max_second_dim - encoded_candidate.size()[1], encoded_candidate.size()[2]).fill_(0)], 1)
        encoded_source = torch.cat([encoded_source,
            torch.cuda.FloatTensor(encoded_source.size()[0], max_second_dim - encoded_source.size()[1], encoded_source.size()[2]).fill_(0)], 1)
        # print("\nEncoded Source Size: ", encoded_source.size())
        # print("Encoded Candidate Size: ", encoded_candidate.size())
        # Build the input to the integrator
        integrator_input_source = torch.cat([encoded_source,
                                      encoded_source.sub(encoded_candidate),
                                      encoded_source * encoded_candidate], 2)
        integrator_input_candidate = torch.cat([encoded_candidate,
                                             encoded_candidate.sub(encoded_source),
                                             encoded_candidate * encoded_source], 2)
        integrated_encodings_source = self._integrator_source(integrator_input_source, source_mask)
        integrated_encodings_candidate = self._integrator_candidate(integrator_input_candidate, candidate_mask)

        # Simple Pooling layers
        # Source
        max_masked_integrated_encodings_source = util.replace_masked_values(
                integrated_encodings_source, source_mask.unsqueeze(2), -1e7)
        max_pool_source = torch.max(max_masked_integrated_encodings_source, 1)[0]
        min_masked_integrated_encodings_source = util.replace_masked_values(
                integrated_encodings_source, source_mask.unsqueeze(2), +1e7)
        min_pool_source = torch.min(min_masked_integrated_encodings_source, 1)[0]
        mean_pool_source = torch.sum(integrated_encodings_source, 1) / torch.sum(source_mask, 1, keepdim=True)
        # Candidate
        max_masked_integrated_encodings_candidate = util.replace_masked_values(
            integrated_encodings_candidate, candidate_mask.unsqueeze(2), -1e7)
        max_pool_candidate = torch.max(max_masked_integrated_encodings_candidate, 1)[0]
        min_masked_integrated_encodings_candidate = util.replace_masked_values(
            integrated_encodings_candidate, candidate_mask.unsqueeze(2), +1e7)
        min_pool_candidate = torch.min(min_masked_integrated_encodings_candidate, 1)[0]
        mean_pool_candidate = torch.sum(integrated_encodings_candidate, 1) / torch.sum(source_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits_source = self._self_attentive_pooling_projection_source(
                integrated_encodings_source).squeeze(2)
        self_weights_source = util.masked_softmax(self_attentive_logits_source, source_mask)
        self_attentive_pool_source = util.weighted_sum(integrated_encodings_source, self_weights_source)
        # Candidate
        self_attentive_logits_candidate = self._self_attentive_pooling_projection_candidate(
            integrated_encodings_candidate).squeeze(2)
        self_weights_candidate = util.masked_softmax(self_attentive_logits_candidate, candidate_mask)
        self_attentive_pool_candidate = util.weighted_sum(integrated_encodings_candidate, self_weights_candidate)

        pooled_representations_source = torch.cat([max_pool_source, min_pool_source, mean_pool_source,
                                                   self_attentive_pool_source], 1)
        pooled_representations_dropped_source = self._integrator_dropout(pooled_representations_source)
        pooled_representations_candidate = torch.cat(
            [max_pool_candidate, min_pool_candidate, mean_pool_candidate, self_attentive_pool_candidate], 1)
        pooled_representations_dropped_candidate = self._integrator_dropout(pooled_representations_candidate)

        logits = self._output_layer(pooled_representations_dropped_source-pooled_representations_dropped_candidate)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        
        if metadata is not None:
            output_dict['words'] = 'asd'

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    # The FeedForward vs Maxout logic here requires a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MTClassifier2':  # type: ignore
        # pylint: disable=arguments-differ
        embedder_params = params.pop("source_field_embedder")
        source_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        embedder_params = params.pop("candidate_field_embedder")
        candidate_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        pre_encode_feedforward_source = FeedForward.from_params(params.pop("pre_encode_feedforward_source"))
        encoder_source = Seq2SeqEncoder.from_params(params.pop("encoder_source"))
        pre_encode_feedforward_candidate = FeedForward.from_params(params.pop("pre_encode_feedforward_candidate"))
        encoder_candidate = Seq2SeqEncoder.from_params(params.pop("encoder_candidate"))
        integrator_source = Seq2SeqEncoder.from_params(params.pop("integrator_source"))
        integrator_candidate = Seq2SeqEncoder.from_params(params.pop("integrator_candidate"))
        integrator_dropout = params.pop("integrator_dropout")

        output_layer_params = params.pop("output_layer")
        if "activations" in output_layer_params:
            output_layer = FeedForward.from_params(output_layer_params)
        else:
            output_layer = Maxout.from_params(output_layer_params)

        elmo = params.pop("elmo", None)
        if elmo is not None:
            elmo = Elmo.from_params(elmo)
        use_input_elmo = params.pop_bool("use_input_elmo", False)
        use_integrator_output_elmo = params.pop_bool("use_integrator_output_elmo", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   source_field_embedder=source_field_embedder,
                   candidate_field_embedder=candidate_field_embedder,
                   embedding_dropout=embedding_dropout,
                   pre_encode_feedforward_source=pre_encode_feedforward_source,
                   encoder_source=encoder_source,
                   pre_encode_feedforward_candidate=pre_encode_feedforward_candidate,
                   encoder_candidate=encoder_candidate,
                   integrator_source=integrator_source,
                   integrator_candidate=integrator_candidate,
                   integrator_dropout=integrator_dropout,
                   output_layer=output_layer,
                   elmo=elmo,
                   use_input_elmo=use_input_elmo,
                   use_integrator_output_elmo=use_integrator_output_elmo,
                   initializer=initializer,
                   regularizer=regularizer)