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


# with open("E:/jiangxinyang/pa_smart_city_nlp/text_classifier/config/mlp_config.json", "r") as fr:
#     config = json.load(fr)

with open(os.path.join(os.path.dirname(os.getcwd()), "data/test_data.txt"), "r", encoding="utf8") as f:
    data = [line for line in f.readlines()]
    inputs = []
    labels = []
    for line in data:
        try:
            x, y = line.strip().split("<SEP>")
            inputs.append(x.strip())
            labels.append(y.strip())
        except:
            print(line)

# text = " ".join([" ".join(jieba.lcut(line)) for line in data])

predictor = Predictor(config)

total = len(labels)
print(set(labels))
corr = 0
for i in range(len(inputs)):
    result = predictor.predict(inputs[i].split(" "))
    if result == labels[i]:
        corr += 1
    else:
        print(inputs[i])
print(corr / total)

