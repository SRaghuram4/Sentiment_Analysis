import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix
import csv
from keras.callbacks import ModelCheckpoint
import re

class TrainModel:
    """
    Class to perform training of LSTM model on provided dara path for sentiment classification on given sentence
    Assumptions:
        a) Train datais supposed to be set in csv format : (n rows and 2 columns)
            a) fishing 1
            b) fresh 1
            c) testing 1

            The frst column is the sentence(string). The second column represents the corresponding sentiment (either 1 or 0)
        b) Test data format should be the same as of train data.

        The data has not been pre-processed. Meaning: no stop words removed. Punctuations are taken care of.
        Remember to 'pip install h5py' before running the code.
    """

    def __init__(self, train_data_path = "\your\path", test_data_path = "\your\path",
                 numOfClasses = 2, glove_path = "\your\glove\path", output_model_path = "\your\path\to\save\model",
                 numpySeed = 1, maxLen = 15):
        """
        Constructor for the class

        :param train_data_path: String. Path to the training data
        :param test_data_path: String. Path to tht testing data
        :param numOfClasses: Integer. e.g 2 as in positive and negative sentiment
        :param glove_path: String. Path to the glove vector to be used
        :param output_model_path: String. Path to the saving the new model
        :return None
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.numOfClasses = numOfClasses
        self.glove_path = glove_path
        self.output_model_path = output_model_path
        np.random.seed(numpySeed)
        self.maxLen = maxLen

    def readData(self, train = False, test = False):
        """
        Function to read the user provided data paths
        :param train: Boolean. If True, read training data using self.train_data_path and returns X_train and Y_train
        :param test: Boolean. If True, read testing data using self.test_data_path and returns X_test and Y_test
        :return: Tupple (nd.array, nd.array). Either (X_train, Y_train) or (X_test, Y_test)
        """

        filename= ''
        if train:
            filename = self.train_data_path
        if test:
            filename = self.test_data_path

        return self.helper_readFile(filename)

    def helper_readFile(self, filename):
        """
        Function to help the readData function actually read the data
        :param filename: String. path to train and test
        :return: Tupple
        """

        phrase = []
        sentiment = []
        try:
            with open(filename) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                for row in csvReader:
                    phrase.append(row[0])
                    sentiment.append(row[1])
        except FileNotFoundError as e:
            print("File not found. Error {0} occured. Please choose a filepath".format(e))
        finally:
            return np.asarray(phrase), np.asarray(sentiment, dtype=int)

    def read_glove_vec(self):
        """
        Function to read pre-trained glove vectors from the disk and return:
            a) words_to_index : a dict e.g: { "a": 0, "aa": 1, "ab": 2...}

            b) index_to_words : a dict e.g {0: "a", 1: "aa", 2: "ab"...}

            c) word_to_vec : dict e.g: {"a": [50 dim array], "aa": [50 dim array]...}
        :return: Tuple(dict, dict, dict)
        """
        word_to_vec_map = []
        try:
            with open(self.glove_path, "r", encoding="utf-8") as f:
                words = set()
                word_to_vec = {}
                for line in f:
                    line = line.strip().split()
                    curr_word = line[0]
                    words.add(curr_word)
                    word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

                i = 1
                words_to_index = {}
                index_to_words = {}

                for w in sorted(words):
                    words_to_index[w] = i
                    index_to_words[i] = w
                    i = i + 1

            return words_to_index, index_to_words, words_to_index
        except FileNotFoundError as e:
            print("File not found. Error {0} occured. Please choose a filepath".format(e))
            return []


    def sentence_to_indices(self, X, word_to_index, maxLen):
        """
        A Function to convert the array of sentences to an array of indices corresponsing to those words in the sentence.
        :param X: -- array of sentences of shape (m, 1)
        :param word_to_index: dict. a dictionary containing the each word mapped to its index
        :param maxLen: integer. max number of words in a sentence
        :return: X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, maxLen)
        """

        m = X.shape[0] # number of training examples

        # Initializing X_indices as a numpy matrix of zeros and the correct shape
        X_indices = np.zeros(shape=(m, maxLen))

        for i in range(m):

            # Converting ith training sentences into lower cases and then stripping and splitting ito words
            sentence_words = X[i].lower().strip().split()
            j = 0

            # Looping over the words of sentences
            for w in sentence_words:
                try:
                    X_indices[i, j] = word_to_index[w]
                except KeyError as e:
                    X_indices[i, j] = word_to_index["unk"]
                j = j + 1
                if j >= maxLen:
                    break
        return X_indices


    def pretrained_embedding_layer(self,word_to_vec_map, word_to_index):
        """
        A function to insert keras embedding before the sentences directly go into hidden layers
        :param word_to_vec_map: dict. a dictionary mapping  words to their GloVe vector representations
        :param word_to_index: dict. a dictionary containing the each word mapped to its index
        :return: embedding layer. pretrained layer Keras instance
        """

        vocab_len = len(word_to_index) + 1 # adding 1 to fit keras embedding
        emb_dim = word_to_vec_map["cucumber"].shape[0] # defining the dim of GloVe vector

        # initializing the embedding matrix with zeros of shape (vocab_len, emb_dim)
        emb_matrix = np.zeros(shape=(vocab_len, emb_dim))

        for word, index in word_to_index.items():
            try:
                emb_matrix[index, 0:len(word_to_vec_map[word])] = word_to_vec_map[word]
            except ValueError as e:
                print("Error inserting word into embedding matrix: ".format(word))

        # defining keras embedding layer with the correct output/input sizes, make it trainable
        embedding_layer = Embedding(vocab_len,emb_dim, trainable=True)

        # building the embedding layer, it is required before setting the weights of the embedding layer
        embedding_layer.build((None,))

        # setting the weights
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer

    def call_maxLen(self, X):
        """
        A function to compute the maximum length of a sentence in the training data
        :param X: training data
        :return: Integer.
        """

        maxLen = -1
        for sent in X:
            k = len(sent.split(" "))
            if maxLen < k:
                maxLen = k

        return maxLen

    def sentimentify(self, input_shape, word_to_vec_map, word_to_index, numberOf_hidden_units = 256,
                     dropout = 0.4, activationDense = "relu", activationLast = "relu"):
        """

        :param input_shape:
        :param word_to_vec_map:
        :param word_to_index:
        :param numberOf_hidden_units:
        :param dropout:
        :param activationDense:
        :param ativationLast:
        :return:
        """
        # defining sentence_indices as input of the grapgh
        sentence_indices = Input(shape=input_shape, dtype="int32")

        # creating the embedding layer
        embedding_layer = self.pretrained_embedding_layer(word_to_vec_map, word_to_index)

        # Propagate through sentence indices
        embeddings = embedding_layer(sentence_indices)

        # Propagating through LSTM layer with 55-dimensional hidden state
        # thr returned output should be batch of sequences
        # 55 -dimesion: maxLen of sentences = 55

        X = Bidirectional(LSTM(numberOf_hidden_units, return_sequences=True))(embeddings)

        # Adding a dropout with prob of 0.5
        X = Dropout(dropout)(X)

        X = LSTM(numberOf_hidden_units)(X)

        # Adding a dropout again
        X = Dropout(dropout)(X)

        # Propagate X through Dense layer with softmax activation to get back 2-D vectors
        X = Dense(1, activation=activationDense)(X)

        # Add a softmax activation
        X = Activation(activationLast) (X)

        # creating model instance which converts sentence_indices into X
        model = Model(sentence_indices, X)

        return model


    def train(self, x_train, y_train, numberOf_hidden_units = 256, NNarchitecture = "BiLSTM", numOfClasses = 2, loss = "mean_squared_error",
              optimizer = "adam", metrics = ['accuracy'], epochs = 500, batch_size = 200, shuffle = True, filePath_to_saveModel = "BiLSTM.hdf5"):


        word_to_index, index_to_words, word_to_vec_map = self.read_glove_vec()

        # calculating the entire length of sentence
        self.maxLen = self.call_maxLen(x_train)

        # generating model from requested architecture
        if NNarchitecture == "BiLSTM":
            model = self.sentimentify(input_shape=(self.maxLen,), word_to_vec_map=word_to_vec_map, word_to_index= word_to_index, numberOf_hidden_units=numberOf_hidden_units)
            print(model.summary())

        # compile the model
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        # transforming training data into the one which is required
        X_train_indices = self.sentence_to_indices(x_train, self.word_to_index, self.maxLen)

        # saving the model
        checkpoint = ModelCheckpoint(filePath_to_saveModel, monitor="val_acc", verbose=1, save_best_only=True, mode='max' )

        #start model training
        model.fit(X_train_indices, epochs=epochs, batch_size=batch_size, shuffle= shuffle, validation_split=0.2)

        model.save(filePath_to_saveModel)

        # Calculate traning accuracy
        loss, acc = model.evaluate(X_train_indices, y_train)
        print("Training Accuracy = {}, Loss = {}".format(acc, loss))

        return model

    def testModel(self, model, X_test, y_test):
        """
        a function to test the model we have trained using BiLSTM
        :param model:
        :param X_test:
        :param y_test:
        :return:
        """

        X_test_indices = self.sentence_to_indices(X_test, self.word_to_index, self.maxLen)

        loss, acc = model.evaluate(X_test_indices, y_test)
        print("Testing Accuracy = {}".format(acc))

    def predictSentiment(self, x_test, model):
        '''
        a function predict sentiment value for user input data
        :param x_test:
        :param model:
        :return:
        '''

        x_test_processed = []
        strip_special_char = re.compile("[^A-Za-z0-9]+")
        for sentence in x_test:
            sentence = sentence.strip().lower()
            sentence = re.sub(strip_special_char, "", sentence).strip()
            x_test_processed.append(sentence)

        x_test_processed = np.asarray(x_test_processed[:])
        print("x_test_processed : {}".format(x_test_processed[0:10]))

        X_test_indices = self.sentence_to_indices(x_test_processed, self.word_to_index, self.maxLen)

        return model.predict(X_test_indices, verbose = 1)


    def computeConfusionMatric(self, prediction, y_actual):
        '''

        :param prediction:
        :param y_actual:
        :return:
        '''

        pred_test2 = []
        for prediction in prediction:
            pred_test2.append(np.argmax(prediction))

        print(confusion_matrix(y_actual, np.asarray(pred_test2)))

        return confusion_matrix(y_actual, np.asarray(pred_test2))

    def print_metrics(self, Y_test_new, pred_test_new):
        '''
        a function to directly compute the metrics from confusion matrix values
        :param Y_test_new:
        :param pred_test_new:
        :return:
        '''
        print("Below are the metrics from the confusion matrix : \n")
        conf_matrix = confusion_matrix(Y_test_new, pred_test_new)
        print("True Negative : {0}".format(conf_matrix[0][0]))
        print("False Negative : {0}".format(conf_matrix[1][0]))
        print("True Positive : {0}".format(conf_matrix[1][1]))
        print("False Negative : {0}".format(conf_matrix[0][1]))
        print("False Positive Rate : {0}".format(conf_matrix[0][1])/(conf_matrix[0][1]+conf_matrix[0][0]))
        print("Accuracy : {0}".format((conf_matrix[0][1]+conf_matrix[1][1])/(conf_matrix[0][1]+conf_matrix[1][1]+
                                                                             conf_matrix[0][0]+conf_matrix[1][0])))
        return conf_matrix


if __name__ == "__main__":

    # define parameters for the model
    numOfClasses = 2
    loss = "mean_square_error"
    optimizer = 'adam'
    metrics = ['mse'][:]
    epochs = 50
    batch_size = 300
    shuffle = True
    numberOf_hidden_units = 256
    train_data_path = "\your\path"
    test_data_path = "\your\path"
    glove_path = '\your\path'

    sentiment = TrainModel(train_data_path, test_data_path, numOfClasses=numOfClasses, glove_path=glove_path)

    # Reading training data from the disk
    X_train, Y_train = sentiment.readData(train=True)

    # Training the model
    model = sentiment.train(x_train=X_train, y_train=Y_train, NNarchitecture='BiLSTM', numberOf_hidden_units=numberOf_hidden_units,
                            numOfClasses=numOfClasses, loss=loss, epochs=epochs, batch_size=batch_size, shuffle = shuffle)

    # Testing the model
    X_test, Y_test = sentiment.readData(test=True)
    sentiment.testModel(model, X_test, Y_test)

    # Prediction
    pred_test = sentiment.predictSentiment(X_test, model)
    print(pred_test[0:10])

    # compute confusion matrix
    cm = sentiment.computeConfusionMatric(pred_test, Y_test)

    # Print the metrics
    pm = sentiment.print_metrics(Y_test, pred_test)
