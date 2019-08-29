import json
import jieba
from predict import Predictor

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", help="config path of model")
args = parser.parse_args()

# 当前目录的上一级目录
with open(os.path.join(os.path.dirname(os.getcwd()), args.config_path), "r") as fr:
    config = json.load(fr)

# with open("E:/jiangxinyang/pa_smart_city_nlp/text_generator/config/seq2seq_bigru_config.json", "r") as fr:
#     config = json.load(fr)

predictor = Predictor(config)
text = jieba.lcut("你们想干嘛")
print(text)
result = predictor.predict(text)
print(result)
