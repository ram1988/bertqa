import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
#import bert
#from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
import json
import pickle
def load_qa_dataset():

    with open(
            './dataset/simplified-nq-train.jsonl') as json_file:
        json_train_records = []
        for i,line in enumerate(json_file):
            data = json.loads(line)
            json_train_records.append(data)
            if i==20000:
                break
        pickle.dump(json_train_records,open("train.pkl","wb"))

    with open(
            './dataset/simplified-nq-test.jsonl') as json_file:
        json_test_records = []
        for i,line in enumerate(json_file):
            data = json.loads(line)
            json_test_records.append(data)
            if i==1000:
                break
        pickle.dump(json_test_records, open("test.pkl","wb"))

def convert_dataset_to_squad_format():
    whole_json_obj = {"version":"v1.0"}
    with open(
            './train.pkl',"rb") as json_file:
        jsonobj = pickle.load(json_file)
        data = []
        for i,record in enumerate(jsonobj):
            print(i)
            obj = convert_each_rcd_to_squad_format(i, record)
            data.append(obj)
        whole_json_obj["data"] = data
    with open('custom_train.json', 'w') as outfile:
        json.dump(whole_json_obj,outfile)


def convert_each_rcd_to_squad_format(idx, record):
    converted_dataset = {}
    context = record["document_text"]
    converted_dataset["title"] = context.split(" ")[0]
    converted_dataset["paragraphs"] = []
    questions = {}
    questions["qas"] = []
    question_list = []
    question = {}
    question["question"] = record["question_text"]
    question["id"] = idx
    answers = []
    for annotation in record["annotations"]:
        answer = {}
        if "long_answer" in annotation and len(annotation["long_answer"].keys())!=0:
            long_answer_record = prepare_answer_record(annotation["long_answer"], context)
            answers.append(long_answer_record)
        if "short_answers" in annotation and len(annotation["short_answers"])!=0:
            short_answers = annotation["short_answers"]
            for short_answer in short_answers:
                short_answer_record = prepare_answer_record(short_answer, context)
                answers.append(short_answer_record)
    question["answers"] = answers
    question["is_impossible"] = False
    question_list.append(question)
    questions["qas"].extend(question_list)
    questions["context"] = context
    converted_dataset["paragraphs"].append(questions)
    return converted_dataset

def prepare_answer_record(answer_obj, document_text):
    answer_start = answer_obj["start_token"]
    answer_end = answer_obj["end_token"]
    text = document_text[answer_start:answer_end]
    return {"text": text, "answer_start": answer_start}

if __name__ == "__main__":
    convert_dataset_to_squad_format()
    # https://medium.com/datadriveninvestor/extending-google-bert-as-question-and-answering-model-and-chatbot-e3e7b47b721a