// Starter configuration for CSE 447/547M A2.
// ==========================================

{
  // Pretrained embedding links
  // ==========================
  //
  // * Sections 1 and 2 *
  // Please use the Google News word2vec 300D pretrained embeddings, which
  // are locally accessible from aziak/attu at:
  //     /cse/web/courses/cse447/19wi/assignments/resources/word2vec/GoogleNews-vectors-negative300.txt.gz
  // and remotely at:
  //     https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz
  //
  // * Section 3 *
  // You are welcome to use other pre-trained embeddings, but just in case,
  // GloVe embeddings are locally accessible from aziak/attu in this
  // directory:
  //     /cse/web/courses/cse447/19wi/assignments/resources/glove/
  // and remotely accessible in:
  //     https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/


  // Stanford Sentiment Treebank configuration
  // =========================================

  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",

  // The test data can be found at:
  // https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt

  "dataset_reader": {
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  }
}
