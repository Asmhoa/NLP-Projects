// Part of this config is taken from https://github.com/allenai/allennlp/blob/088f0bb/training_config/bidirectional_language_model.jsonnet

{
    "vocabulary": {
        "tokens_to_add": {
            "tokens": ["<S>", "</S>"]
        },
        "min_count": {
            "tokens": 3
        }
    },
    "dataset_reader": {
        "type": "simple_language_modeling",
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
    },
    "train_data_path": "/homes/iws/amols2/NLP/tokens/1b_benchmark.train.tokens",
    "model": {
        "type": "language_model",
        "text_field_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "embedding_dim": 32
                }
            }
        },
        "contextualizer": {
            "type": "lstm",
            "input_size": 32,
            "hidden_size": 256
        }
    },
    "iterator": {
        "type": "basic"
    },
    "trainer": {
        "type": "default",
        "optimizer": {
            "type": "adam"
        },
	"cuda_device": 0,
    },
}
