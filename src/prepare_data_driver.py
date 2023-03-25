import os
import sys
PACKAGE_ROOT = os.path.abspath("")
print(PACKAGE_ROOT)
sys.path.insert(0, PACKAGE_ROOT)

from preprocessing.prepare_connl_data import PrepareConnlData
from commons import constants as C

# train data
PrepareConnlData(C.TXT_CONNL_TRAIN_PATH, "train.csv").run()
PrepareConnlData(C.TXT_CONNL_VALID_PATH, "valid.csv").run()
PrepareConnlData(C.TXT_CONNL_TEST_PATH, "test.csv").run()
