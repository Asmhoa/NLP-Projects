{
    "dataset_reader": {
		"type": "pos_tagger_read",
		"token_indexers": {
			"tokens": {
				"type": "single_id",
				"lowercase_tokens": true
			},
			"elmo": {
				"type": "elmo_characters"
			}
		}
	},
    "train_data_path": "./a4-data/a4-train.conllu",
	"validation_data_path": "./a4-data/a4-dev.conllu",
    "model": {
        "type": "structured_perceptron_tagger",
		"text_field_embedder": {
			"tokens": {
				"type": "embedding",
				"pretrained_file": "/cse/web/courses/cse447/19wi/assignments/resources/glove/glove.840B.300d.txt.gz",
				"trainable": true,
				"embedding_dim": 300
			},
			"elmo": {
				"type": "elmo_token_embedder",
				"options_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_small_options.json",
				"weight_file": "/cse/web/courses/cse447/19wi/assignments/resources/elmo/elmo_small_weights.hdf5",
				"do_layer_norm": false,
				"dropout": 0.5
			}
		},
		"encoder": {
			"type": "stacked_bidirectional_lstm",
			"input_size": 556,
			"hidden_size": 128,
			"num_layers": 2
			
		}
    },
	"iterator": {
		"type": "bucket",
		"sorting_keys": [["tokens", "num_tokens"]],
		"batch_size": 128
	},
    "trainer": {
        "num_epochs": 50,
        "patience": 10,
        "cuda_device": 0,
        "optimizer": {
            "type": "adagrad",
			"lr": 0.01
        }
    },
}
