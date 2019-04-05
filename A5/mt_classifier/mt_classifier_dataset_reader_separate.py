from typing import Dict, Tuple, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("mt_classifier_reader_separate")
class MTClassifierDatasetReaderSeparate(DatasetReader):
    """
    Reads a file in the MT Classifier assignment format.

    Parameters
    ----------
    source_language : ``str``, optional, (default = 'de_core_news_sm')
        The name of the spaCy model used to tokenize the source sentences.  
        Models can be found here <https://spacy.io/models/>.
    candidate_language : ``str``, optional (default = 'en_core_web_sm')
        The name of the spaCy model uwed to tokenize the candidate sentences.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
            The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 source_language: str = 'de_core_news_sm',
                 candidate_language: str = 'en_core_web_sm',
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._source_tokenizer = SpacyWordSplitter(language=source_language)
        self._candidate_tokenizer = \
                SpacyWordSplitter(language=candidate_language)


    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)

        with open(file_path, 'r') as mt_file:
            logger.info("Reading MT instances dataset at: %s", file_path)

            for line in mt_file:
                if not line:
                    continue
                else:
                    inputs = line.strip().split("\t")
                    source = inputs[0]
                    candidate = inputs[1]
                    label = inputs[2]

                    yield self.text_to_instance(source, candidate, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         source: str,
                         candidate: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        source : ``str``, required
            The translation's source sentence.
        candidate : ``str``, required
            The translation candidate.
        label : ``str``, optional (default = None)
            Whether the candidate is human- or machine-translated, if known.
        """
        fields: Dict[str, Field] = {}

        source_tokens = self._source_tokenizer.split_words(source)
        candidate_tokens = self._candidate_tokenizer.split_words(candidate)

        fields["source"] = TextField(source_tokens, self._token_indexers)
        fields["candidate"] = TextField(candidate_tokens, self._token_indexers)
        
        if label:
            fields["label"] = LabelField(label)
        
        fields["metadata"] = MetadataField({"source": source, 
                                            "candidate": candidate})
        return Instance(fields)
