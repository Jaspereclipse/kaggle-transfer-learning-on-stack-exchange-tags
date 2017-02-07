from _init_ import config
import tensorflow as tf

# # takes memory
# word2id = load_pickle(config['vocabulary'])
# id2word = {v: k for k, v in word2id.iteritems()}
# df_sent = load_pickle(config['training_data'])

flags = tf.app.flags
# directories
flags.DEFINE_string("save_path", config['save_path'], "save path")
# files
flags.DEFINE_string("train_data", config['training_data'], "Doc2Vec training data")
flags.DEFINE_string("vocabulary", config['vocabulary'], "Vocabulary")
flags.DEFINE_string("vocabulary_counts", config['vocabulary_counts'], "Vocabulary counts")
# model params
flags.DEFINE_integer("vocab_size", None, "vocabulary size") # include NULL, UNK
flags.DEFINE_integer("num_paragraphs", None, "number of paragraphs in the training set")
flags.DEFINE_integer("doc_embedding_size", 200, "embedding size for paragraph")
flags.DEFINE_integer("word_embedding_size", 100, "embedding size for word")
flags.DEFINE_integer("batch_size", 128, "number of training samples per batch")
flags.DEFINE_integer("num_neg_samples", 100, "number of negative samples per training sample")
flags.DEFINE_integer("epochs", 15, "number of epochs to train")
flags.DEFINE_float("learning_rate", 0.2, "learning rate")
flags.DEFINE_string("mode", "concat", "ways to combine paragraph with word embeddings, e.g. [concat|sum|average]")
flags.DEFINE_integer("window", 5, "number of preceding words used to predict next word")
# train param
flags.DEFINE_integer("threads", 8, "number of threads used in training")
flags.DEFINE_integer("stats_interval", 10, "Print statistics every N seconds")
flags.DEFINE_integer("summary_interval", 10, "Save training summary to file every N seconds")
flags.DEFINE_integer("checkpoint_interval", 600, "Save model parameters every N seconds")
FLAGS = flags.FLAGS

class Options(object):
    """
    Options used by doc2vec model
    """

    def __init__(self):
        self.save_path = FLAGS.save_path
        self.train_data = FLAGS.train_data
        self.vocabulary = FLAGS.vocabulary
        self.vocabulary_counts = FLAGS.vocabulary_counts
        self.wd_emb_dim = FLAGS.word_embedding_size
        self.ph_emb_dim = FLAGS.doc_embedding_size
        self.vocab_size = FLAGS.vocab_size
        self.num_paragraphs = FLAGS.num_paragraphs
        self.batch_size = FLAGS.batch_size
        self.num_samples = FLAGS.num_neg_samples
        self.window = FLAGS.window
        self.mode = FLAGS.mode
        if self.mode in ["sum", "average"]: # sum/average
            assert self.wd_emb_dim == self.ph_emb_dim,\
            "If not concatenated, paragraph embeddings should have the same size as word embedding's"
            self.input_embed_dim = self.ph_emb_dim
        else:
            self.input_embed_dim = self.window * self.wd_emb_dim + self.ph_emb_dim
        self.epochs = FLAGS.epochs
        self.learning_rate = FLAGS.learning_rate
        self.threads = FLAGS.threads
        self.stats_interval = FLAGS.stats_interval
        self.summary_interval = FLAGS.summary_interval