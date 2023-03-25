from commons import constants as C
import pandas as pd

class MemoryTaggerConnlPreprocessor:
    def __init__(self):
        self.file_mapper = {
            "train": C.CSV_CONLL_TRAIN_PATH, 
            "valid": C.CSV_CONLL_VALID_PATH, 
            "test": C.CSV_CONLL_TEST_PATH
        }

    def get_preprocessed_data(self, split_type):
        df = pd.read_csv(self.file_mapper[split_type])
        agg_df = df.groupby(["sentence_id"]).agg( word_list=pd.NamedAgg(column="token", aggfunc=list), 
                                            tag_list=pd.NamedAgg(column="tag", aggfunc=list) ).reset_index()
        X = list(agg_df.word_list.values)
        y = list(agg_df.tag_list.values)

        return X, y