import json
from collections import namedtuple
import pandas as pd

FILE_SOURCE = "E:/SchoolCourses/6490-DocumentAnalysis/Assignment/Assignment 2/question_2-2_data/train.json"
Doc = namedtuple('Doc', 'id label sentence')

def readJson(file_source) :
    with open(file_source, 'r') as myF:
        data = json.load(myF)
    doc_list = []
    for dic in data :
        doc_list.append(Doc(id = dic["ID"], label = dic["Label"], sentence = dic["Sentence"]))
    return pd.DataFrame(doc_list, columns = Doc._fields)

dataset = readJson(FILE_SOURCE)

print(dataset["sentence"][0:21])