import os
import sys
PACKAGE_ROOT = os.path.abspath("")
print(PACKAGE_ROOT)
sys.path.insert(0, PACKAGE_ROOT)
from data_preparation.prepare_connl_data import PrepareConnlData
from commons import constants as C

# train data
PrepareConnlData(C.CONNL_TRAIN_PATH, "train.csv").run()
PrepareConnlData(C.CONNL_VALID_PATH, "valid.csv").run()
PrepareConnlData(C.CONNL_TEST_PATH, "test.csv").run()
