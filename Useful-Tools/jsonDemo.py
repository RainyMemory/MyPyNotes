import json
from collections import namedtuple
import pandas as pd

FILE_SOURCE = "./Assignment 2/question_2-2_data/train.json"
Doc = namedtuple('Doc', 'id label sentence')

def readJson(file_source) :
    with open(file_source, 'r') as myF:
        data = json.load(myF)
    doc_list = []
    for dic in data :
        if "Label" in dic :
            doc_list.append(Doc(id = dic["ID"], label = dic["Label"], sentence = dic["Sentence"]))
        else :
            doc_list.append(Doc(id = dic["ID"], label = None, sentence = dic["Sentence"]))
    return pd.DataFrame(doc_list, columns = Doc._fields)

if __name__ == "__main__":
    dataset = readJson(FILE_SOURCE)
    print(dataset["sentence"][0:21])