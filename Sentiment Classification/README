# CSE 447/547M W19: A2

## Included files

* skeleton.jsonnet: starter configuration; contains pointers to pretrained word embedding files and the Stanford Sentiment Treebank dataset.

* plaintext_reader.py and predictor.py: if you want to try new input examples against your model that aren't in the SST format, create a file with one input sentence per line and run via:

    allennlp predict \
        <model_path> <input_path> \
        --include-package <package_name> \
        --predictor sentiment_predictor \
        --overrides "{dataset_reader: {type: 'sentiment_plaintext_reader'}}"


## Tips for development

* Writing unit tests may make your model development process easier. Here's part of a tutorial that describes creating a test class: https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt1.md#step-four-write-your-model

  You can then use pytest (https://docs.pytest.org/en/latest/usage.html) to run the tests you create.

* To run allennlp train (or one of the other commands) with your model, you will need to include the package containing your model (and the reader/predictor provided here) via the --include-package flag. (AllenNLP-provided classes, such as the SST reader, are already included by default.)
