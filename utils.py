from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def max_margin_loss(y_true, y_pred):
    return K.sum(K.maximum(0., y_true*(1-y_pred)+(1-y_true)*(1+y_pred)))


def my_accuracy(y_true, y_pred, threshold=0):
    threshold = K.cast(threshold, y_pred.dtype)
    y_pred = K.cast(y_pred > threshold, y_pred.dtype)#y_pred>0设为1，<=0设为0
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def get_bert_input_data(data, labels, Id2Data, max_seq_len, tokenizer):
    
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "label": []
    }
    assert len(data) == len(labels)
    for i in range(len(data)):
        if len(Id2Data)==0:
            sentence = data[i]
        else:
            sentence = Id2Data[data[i][0]][0]
        
        input_ids = tokenizer.encode(
            sentence,  
            add_special_tokens=True,  
            max_length=max_seq_len,  
        )
        sentence_length = len(input_ids)
        input_ids = pad_sequences([input_ids],
                                  maxlen=max_seq_len,
                                  dtype="long",
                                  value=0,
                                  truncating="post",
                                  padding="post")
        input_ids = input_ids[0]
        attention_mask = [1] * sentence_length + [0] * (max_seq_len - sentence_length)

        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["label"].append(labels[i])

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = dataset_dict["label"]
    return x, y