import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from commons import constants as C

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