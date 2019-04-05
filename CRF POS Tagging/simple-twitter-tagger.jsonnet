{
    "dataset_reader": {
		"type": "twitter_tagger_read"
	},
    "train_data_path": "./en-ud-tweet-train.conllu",
	"validation_data_path": "./en-ud-tweet-dev.conllu",
    "model": {
        "type": "simple_tagger",
		"text_field_embedder": {
			"type": "basic",
			"token_embedders": {
				"tokens": {
					"type": "embedding",
					"pretrained_file": "/cse/web/courses/cse447/19wi/assignments/resources/glove/glove.840B.300d.txt.gz",
					"trainable": true,
					"embedding_dim": 300
				}
			}
		},
		"encoder": {
			"type": "stacked_bidirectional_lstm",
			"input_size": 300,
			"hidden_size": 256,
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