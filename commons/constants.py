import os

HOME_DIR = r"C:\Users\sharm\Documents\nlp_assignment\ner-bilstm-crf"
INPUT_DATA_DIR = os.path.join(HOME_DIR, "inputs")
OUTPUT_DIR = os.path.join(HOME_DIR, "outputs")

CSV_CONLL_TRAIN_PATH = os.path.join(INPUT_DATA_DIR, "train.csv")
CSV_CONLL_VALID_PATH = os.path.join(INPUT_DATA_DIR, "valid.csv")
CSV_CONLL_TEST_PATH = os.path.join(INPUT_DATA_DIR, "test.csv")

TXT_CONNL_TRAIN_PATH = os.path.join(INPUT_DATA_DIR, "en_train.conll")
TXT_CONNL_VALID_PATH = os.path.join(INPUT_DATA_DIR, "en_dev.conll")
TXT_CONNL_TEST_PATH = os.path.join(INPUT_DATA_DIR, "en_test.conll")