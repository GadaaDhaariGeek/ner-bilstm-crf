import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from models.memory_tagger import MemoryTagger

def generate_simple_word_features(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(), word.isalpha()])



class FeatureTransformer(BaseEstimator, TransformerMixin):
    """class to enhance our simple features on the one hand by memory and on the other hand by using context information.
    """
    def __init__(self):
        self.memory_tagger = MemoryTagger()
        self.tag_encoder = LabelEncoder()
        self.pos_encoder = LabelEncoder()

    def fit(self, X, y=None):
        """ fit method

        Args:
            X (list of list): _description_
            y (_type_): _description_
        """
        # all the words as a single list
        words = X["token"].values.tolist()
        # all the tags as single list 
        tags = X["tag"].values.tolist()
        # fit memory tagger
        self.memory_tagger.fit([words], [tags])
        # encode tags
        self.tag_encoder.fit(tags)

    def transform(self, X, y=None):
        # def pos_default(p):
        #     if p in self.pos:
        #         return self.pos_encoder.transform([p])[0]
        #     else:
        #         return -1
        
        # take all the words 
        words = X["token"].values.tolist()
        out = []
        for i in range(len(words)):
            # print(i)
            # for each word w
            w = words[i]
            if i < len(words) - 1:
                wp = self.tag_encoder.transform(self.memory_tagger.predict([[words[i+1]]]))[0]
                # posp = pos_default(pos[i+1])
            else:
                wp = self.tag_encoder.transform(['O'])[0]
                # posp = pos_default(".")
            if i > 0:
                if words[i-1] != ".":
                    wm = self.tag_encoder.transform(self.memory_tagger.predict([[words[i-1]]]))[0]
                    # posm = pos_default(pos[i-1])
                else:
                    wm = self.tag_encoder.transform(['O'])[0]
                    # posm = pos_default(".")
            else:
                # posm = pos_default(".")
                wm = self.tag_encoder.transform(['O'])[0]
            # out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
            #                      self.tag_encoder.transform(self.memory_tagger.predict([w]))[0],
            #                      pos_default(p), wp, wm, posp, posm]))
            out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                                 self.tag_encoder.transform(self.memory_tagger.predict([[w]]))[0],
                                 wp, wm]))
        return out

