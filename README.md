# kaggle-transfer-learning-on-stack-exchange-tags

# Project Name

Predicting topics of Stack Exchange questions based on the context.

## Overview

This project compares different features (tf-idf, doc2vec) and different classification models (Naive Bayes, SVM) in the task of predicting related topic for questions being asked in the Stack Exchange communities.

I implemented my own version of doc2vec model in [TensorFlow r0.12](https://www.tensorflow.org/) mimicing the word2vec example in TensorFlow's [tutorial](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py). A GPU version is also implemented for faster training.

## Usage

To use the doc2vec model, you may need to create your own version of batch generation function since the current one is only used for the purpose of the project. 

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

The project originates from a Kaggle competition: [Transfer Learning on Stack Exchange Tags](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags)

## License

Apache License 2.0
