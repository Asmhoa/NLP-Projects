// Derived from https://github.com/kushalchauhan98/bcn-cnn-text-classification
{
  "dataset_reader":{
    "type": "mt_classifier_reader",
    "lazy":true,
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-multilingual-cased",
          "do_lowercase": false,
          "use_starting_offsets": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
    },
	"train_data_path": "a5-data/de_en.train",
	"validation_data_path": "a5-data/de_en.dev",
  "model": {
    "type": "mt_classifier",
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
        },
        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-multilingual-cased"
            }
        }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": 768,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 2],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 100
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 10,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0005
    }
  }
}
