import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras.utils import pad_sequences, to_categorical
from preprocessing.crf_preprocessor import sent2labels
from utils.utils import SentenceGetter

class LSTMSoftmaxPreprocessor:

    def __init__(self):
        pass

    def inference(self, model, X, idx2tag, sentences):
        # get the predictions
        # y_pred is of shape (-1, max_len, ntags)
        y_pred = model.predict(X)
        # get the tag corresponding to maximum value for a particular word
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = np.vectorize(idx2tag.get)(y_pred)
        i = 0
        final_preds = []
        while i < y_pred.shape[0]:
            final_preds.append(list(y_pred[i][:len(X[i])]))
            i += 1
        y_true = [sent2labels(s) for s in sentences]

        # changing to MLB for calculating metrics report
        mlb = MultiLabelBinarizer()
        yv = mlb.fit_transform(y_true)
        yp = mlb.transform(y_pred)

        # calculate report
        report = classification_report(y_true=yv, y_pred=yp, output_dict=True, target_names=mlb.classes_)
        report_df = pd.DataFrame(report).transpose()
        return report_df

    def inference_one_sample(self, model, X, y, i, words, tags):
        # i = 15
        p = model.predict(np.array([X[i]]))
        # p.shape
        p = np.argmax(p, axis=-1)
        actual_p = np.argmax(y[i], axis=-1)
        # p.shape
        print("{:15} ({:20}): {}".format("Word", "True", "Pred"))
        for w, ap, pred in zip(X[i], actual_p, p[0]):
            print("{:15} ({:20}): {}".format(words[w], tags[ap], tags[pred]))


    def get_sentences(self, df):
        return SentenceGetter(df).sentences

    def get_words(self, df):
        words = list(set(df["token"].values))
        words.append("UNK")
        words.append("ENDPAD")
        return words
    
    def get_tags(self, df):
        tags = list(set(df["tag"].values))
        return tags 
    
    def get_max_length(self, sentences):
        max_length = len(max(sentences, key=len))
        return max_length
    
    def get_indices(self, words, tags):
        word2idx = {w: i for i, w in enumerate(words)}
        idx2word = {i: w for w, i in word2idx.items()}
        tag2idx = {t: i for i, t in enumerate(tags)}
        idx2tag = {i:t for t, i in tag2idx.items()}

        return word2idx, idx2word, tag2idx, idx2tag

    def preprocess(self, sentences, word2idx, tag2idx, max_len, n_tags, n_words):
        X = [[word2idx.get(w[0], word2idx["UNK"]) for w in s] for s in sentences]
        # nwords-1 is the index of 'ENDPAD'
        X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
        y = [[tag2idx[w[1]] for w in s] for s in sentences]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
        y = [to_categorical(i, num_classes=n_tags) for i in y]
        return X, y
