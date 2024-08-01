import jieba
import jieba.posseg
import re
import numpy as np

def get_ID2Data(dataset):
    Id2Data={}
    for i in range(len(dataset["texts"])):
        data = dataset.iloc[i]
        s = data["texts"]
        label = np.array(data[1:-1]).astype('float32')
        index = data["sentence_index"]
        Id2Data[index]=[s,label]
    return Id2Data

def pre_classifier_data(dataset):
    sentences = []
    labels = []
    sentences_emd = []
    for i in range(len(dataset["texts"])):
        data = dataset.iloc[i]
        sentences.append(data["texts"])
        labels.append(np.array(data[1:-1]).astype('float32'))
    labels = np.array(labels)
    return sentences,labels