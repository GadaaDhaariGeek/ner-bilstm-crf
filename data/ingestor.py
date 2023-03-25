from datetime import datetime, date, timedelta
from time import strftime
import os
import joblib

from commons import constants as C

class DataSaver:
    def __init__(self):
        pass

    def save_report(self, report_df, model_type, data_split_type):
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        FILE_NAME = f"{model_type}-model-{data_split_type}-data-"+now.strftime("%d-%m-%Y-%H-%M-%S")+".csv"
        FILE_PATH = os.path.join(C.METRICS_OUTPUT_DIR, FILE_NAME)
        report_df.to_csv(
            path_or_buf=FILE_PATH,
            index=True
        )

    def save_model(self, model, model_type):
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        FILE_NAME = f"{model_type}-model-"+now.strftime("%d-%m-%Y-%H-%M-%S")+".pkl"
        FILE_PATH = os.path.join(C.MODELS_OUTPUT_DIR, FILE_NAME)
        joblib.dump(model, FILE_PATH)

class DataLoader:
    def __init__(self):
        pass

    def load_model(self, path):
        model = joblib.load(path)
        return model
