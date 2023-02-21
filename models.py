# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    # add nn.Module next to SentimentClassifier (double super class)
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, input_size, hidden_size, output_size, embeddings: WordEmbeddings):
        super(NeuralSentimentClassifier, self).__init__()
        self.V = nn.Linear(input_size, hidden_size)
        self.g = nn.Tanh()
        self.W = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=0)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.word_embeddings = embeddings
        self.embeddings = embeddings.get_initialized_embedding_layer()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.V(x)
        x = self.g(x)
        x = self.W(x)
        x = self.log_softmax(x)
        return x

    def predict(self, ex_words: List[str]) -> int:
        """
                Makes a prediction on the given sentence
                :param ex_words: words to predict on
                :return: 0 or 1 with the label
                """

        embedding_list = []
        for word in ex_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            if word_index == -1:
                word_index = 1
            index = torch.LongTensor([word_index])
            embedding_list.append(self.embeddings(index)[0])
        mean = torch.mean(torch.stack(embedding_list), dim=0)
        log_probs = self.forward(mean)
        prediction = torch.argmax(log_probs)
        return prediction.item()


    def form_input(self, ex_words: List[str]) -> torch.Tensor:
        """
        :param x: a [num_samples x inp] numpy array containing input data
        :return: a [num_samples x inp] Tensor
        """
        embedding_list = []
        for word in ex_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            if word_index == -1:
                word_index = 1
            index = torch.LongTensor([word_index])
            embedding_list.append(self.embeddings(index)[0])
        mean = torch.mean(torch.stack(embedding_list), dim=0)
        return mean



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    input_size = word_embeddings.get_embedding_length()
    # batch_size =
    hidden_size = 5
    output_size = 2
    num_epochs = 15
    initial_learning_rate = 0.001

    ffnn = NeuralSentimentClassifier(input_size, hidden_size, output_size, word_embeddings)

    optimizer = optim.Adam(ffnn.parameters(), initial_learning_rate)
    Loss = nn.CrossEntropyLoss()

    for epoch in range(0, num_epochs):
        np.random.shuffle(train_exs)
        total_loss = 0.0
        for ex in train_exs:
            x = ffnn.form_input(ex.words)
            y_onehot = torch.zeros(2)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(ex.label, dtype=np.int64)), 1)
            ffnn.zero_grad()
            log_probs = ffnn.forward(x)
            loss = Loss(log_probs, y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return ffnn




