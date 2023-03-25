
# import numpy as np
import os
import pandas as pd
from time import time, gmtime
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import SentenceGetter
from commons import constants as C

class CRFPreprocessor:
    def __init__(self):
        self.time_prefix = gmtime()
    
    def word2features(self, sent, i):
        # print("sent: ")
        # print(sent)
        word = sent[i][0]
        # print("word", word, "type: ", type(word) )

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i-1][0]
            # print("word1", word1, "type: ", type(word1) )
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            # print(sent[i+1])
            # print("word1", word1, "type: ", type(word1) )
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        try:
            return [self.word2features(sent, i) for i in range(len(sent))]
        except:
            print("senty:")
            print(sent)

    def sent2labels(self, sent):
        return [label for _, label in sent]

    def sent2tokens(self, sent):
        return [token for token, _ in sent]
    
    def get_sentences(self, df):
        return SentenceGetter(df).sentences
    
    def preprocess(self, sentences):
        X = [self.sent2features(s) for s in sentences]
        y = [self.sent2labels(s) for s in sentences]
        return X, y
    
    def inference(self, model, X, y):
        # get the predictions
        # y_pred is of shape (-1, max_len, ntags)
        y_pred = model.predict(X)
        # print(f"Shape of y_pred: {len(y_pred)}")
        # get the tag corresponding to maximum value for a particular word
        # y_pred = np.vectorize(idx2tag.get)(y_pred)
        # y_true = np.vectorize(idx2tag.get)(y)

        # y_true = [label[:len(sentences[idx])] for idx, label in enumerate(y_true)]
        # y_pred = [label[:len(sentences[idx])] for idx, label in enumerate(y_pred)]

        # changing to MLB for calculating metrics report
        mlb = MultiLabelBinarizer()
        yv = mlb.fit_transform(y)
        yp = mlb.transform(y_pred)

        # calculate report
        report = classification_report(y_true=yv, y_pred=yp, output_dict=True, target_names=mlb.classes_)
        report_df = pd.DataFrame(report).transpose()
        return report_df