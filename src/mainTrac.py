import spacy
from spacymoji import Emoji

import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, SpatialDropout1D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.regularizers import L1L2
from preprocessTrac import create_data_task, create_test_data, prepocess_organizers_dataset
from utils import load_obj

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 250
EPOCHS = 50


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average attention mechanism from:
        Zhou, Peng, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao and Bo Xu.
        “Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.”
        ACL (2016). http://www.aclweb.org/anthology/P16-2034
    How to use:
    see: [BLOGPOST]
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]

        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True)  # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def lstm_simple_binary_attent(embedding_matrix):
    # opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    d = 0.5
    rnn_units = 30
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(d))
    model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True,
                                 recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))  # , dropout=d, recurrent_dropout=rd)))
    model.add(AttentionWeightedAverage())
    model.add(Dropout(d))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1])
    return model


if __name__ == '__main__':

    CREATE_TRAIN_DATA = True
    CREATE_TEST_DATA = True

    # csv to tsv conversion
    add_path = "../data/trac2/input/2/trac2_eng_train.csv"
    train_file = "../data/trac2/output/trac2_train.tsv"
    prepocess_organizers_dataset(add_path, train_file)

    add_path = "../data/trac2/input/2/trac2_eng_dev.csv"
    testing_path = "../data/trac2/output/trac2_test.tsv"
    prepocess_organizers_dataset(add_path, testing_path)

    # pre-training pre-processing
    # embedding file
    emb_path = "../data/model/model_swm_300-6-10-low.w2v"
    vocab_path = "../data/trac2/input/2/trac2_eng_train.csv"   #? how to generate this one for new training and test data
    word_index_path = "../data/trac2/input/2/word_index_train.csv"  #? same as above
    # generate below ones
    word_index_pkl_path = "../data/output/word_index_train.pkl"
    embedding_matrix_path = "../data/output/embedding_matrix.pkl"
    data_path = "../data/output/trac2.pkl"
    labels_path = "../data/output/labels_trac2.pkl"

    nlp = None
    if CREATE_TRAIN_DATA:
        nlp = spacy.load('en')
    emoji = Emoji(nlp)
    nlp.add_pipe(emoji, first=True)
    create_data_task(
        train_file,
        vocab_path,
        word_index_pkl_path,
        embedding_matrix_path,
        emb_path,
        data_path,
        labels_path,
        word_index_path,
        nlp)

    print("NLP Loaded")
    word_index = load_obj(word_index_pkl_path)
    print(len(word_index))
    print("Word Index Loaded")
    embedding_matrix = load_obj(embedding_matrix_path)
    print(embedding_matrix.shape)
    print("Embbeding Matrix Created")
    data = load_obj(data_path)

    # TEST
    data_path = "../data/trac2/output/data_test.pkl"
    ids_test_path = "../data/trac2/output/ids_test.pkl"
    result_path = "../data/trac2/output/task_result.csv"

    if CREATE_TEST_DATA:
        if not nlp:
            nlp = spacy.load('en')
            emoji = Emoji(nlp)
            nlp.add_pipe(emoji, first=True)

        create_test_data(data_path, ids_test_path, testing_path, word_index, nlp)

    labels = load_obj(labels_path)

    x_train = data
    y_train = labels

    print("Training Set Processed")

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')

    callbacks_list = [earlystop]

    # train the model
    model = lstm_simple_binary_attent(embedding_matrix)
    print(model.summary())
    class_weight = {0: 4.,
                    1: 1.,
                    2: 1.5}

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        # validation_split=0.15,
                        callbacks=callbacks_list,
                        shuffle=True,
                        class_weight=class_weight)

    label_test_dict = {0: "OAG",
                       1: "CAG",
                       2: "NAG"}

    print("Testing....")

    data_test = load_obj(data_path)
    print(data_test.shape)
    ids_test = load_obj(ids_test_path)

    i = 0

    out = open(result_path, "w")

    while i < len(ids_test):
        prediction = model.predict(np.array([data_test[i]]))
        predicted_label = np.argmax(prediction[0])
        out.write(ids_test[i] + "," + str(label_test_dict[predicted_label]) + "\n")
        i += 1
    out.close()
