import os
import sys
import copy

import pandas as pd

PACKAGE_ROOT = os.path.abspath("")
print(PACKAGE_ROOT)
sys.path.insert(0, PACKAGE_ROOT)

from commons import constants as C
# print(C.CONNL_TEST_PATH)

class PrepareConnlData:
    """
    Class to convert the data from connl format to dataframe.
    This class expects the data to be in a specific format.
    """
    def __init__(self, input_file_path, output_file_name):
        self.input_file_path = input_file_path
        self.output_file_name = output_file_name


    def _prepare_data(self):
        all_sentences = []
        with open(self.input_file_path, "r", encoding="latin-1") as fr:
            for line in fr:
                temp = copy.deepcopy(line).strip()
                if temp != "":
                    if temp.startswith("# id"):# and temp.endswith("domain=en"):
                        sentence_id = temp.split()[2]
                        continue
                    else:
                        all_sentences.append({
                            "sentence_id": sentence_id, 
                            "token": temp.split()[0],
                            "tag": temp.split()[3]
                        })
        self.df = pd.DataFrame(all_sentences)

    def _save_data(self):
        self.df.to_csv(os.path.join(C.DATA_DIR, self.output_file_name), index=False)

    def run(self):
        self._prepare_data()
        self._save_data()

