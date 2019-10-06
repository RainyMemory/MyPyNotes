from jsonDemo import readJson
import csv

CSV_PATH = "./Assignment 2/models/fasttext_model_file_q2-2/fasttext_model_file_q2-2predictions.csv"
TAR_CSV_PATH = "./Assignment 2/models/fasttext_model_file_q2-2/fasttext_model_file_q2-2predictions_format.csv"
JSON_PATH = "./Assignment 2/question_2-2_data/test.json"

def read_csv(path) :
    content = []
    with open(path, 'r') as csv_f :
        text = csv.reader(csv_f)
        for line in text :
            content.append(line)
        csv_f.close()
    return content

def add_csv(path, infolist) :
    with open(path, 'w+') as csv_f :
        for info in infolist :
            csv_f.write(info)
    csv_f.close()

if __name__ == "__main__":
    predict_answer = read_csv(CSV_PATH)
    predict_ids = readJson(JSON_PATH)["id"]
    infolist = []
    infolist.append("id,category\n")
    for index in range(0, len(predict_answer)) :
        infolist.append(predict_ids[index] + "," + predict_answer[index][0] + "\n")
    add_csv(TAR_CSV_PATH, infolist)