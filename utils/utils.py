from commons import constants as C
import pandas as pd

def get_connl_data(split_type):
    file_mapper = {
        "train": C.CSV_CONLL_TRAIN_PATH, 
        "valid": C.CSV_CONLL_VALID_PATH, 
        "test": C.CSV_CONLL_TEST_PATH
    }
    df = pd.read_csv(file_mapper[split_type], keep_default_na=False)
    return df


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [[w, t] for w, t in zip(s["token"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None