import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Model,load_model,Sequential
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input,Concatenate,Lambda,Dot,Activation

def Create_Encoder_Bert_model(MODEL_NAME, MAX_SEQUENCE_LENGTH):
    input_ids_s=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='input_ids_s')
    attention_mask_s=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='attention_mask_s')
    input_ids_d=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='input_ids_d')
    attention_mask_d=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='attention_mask_d')
    input_ids_o=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='input_ids_o')
    attention_mask_o=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32,name='attention_mask_o')
    
    bertModel = TFBertModel.from_pretrained(MODEL_NAME)
      
    reason_emd_s = bertModel.bert(input_ids_s, attention_mask = attention_mask_s)[1]
    reason_emd_d = bertModel.bert(input_ids_d, attention_mask = attention_mask_d)[1]
    reason_emd_o = bertModel.bert(input_ids_o, attention_mask = attention_mask_o)[1]
   
    output_s = Lambda(lambda x: x,name='output_s')(reason_emd_s)
    output_d = Lambda(lambda x: x,name='output_d')(reason_emd_d)
    output_o = Lambda(lambda x: x,name='output_o')(reason_emd_o)
    
    doc_sd_cos=Dot(axes=(-1, -1), normalize=True)([output_s,output_d])
    doc_do_cos=Dot(axes=(-1, -1), normalize=True)([output_d,output_o])
    doc_sub=Lambda(lambda x: x[0]-x[1],name='doc_sub')([doc_sd_cos,doc_do_cos])
    
    model = Model(inputs=[input_ids_s, input_ids_d, input_ids_o,
                          attention_mask_s, attention_mask_d, attention_mask_o], 
                  outputs=[doc_sub])

    return model

def aspects_CNN_classifier(input_size,num_labels,all_encoder_models,MAX_SEQUENCE_LENGTH):
    
    input_ids=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32)
    attention_mask=Input(shape=(MAX_SEQUENCE_LENGTH),dtype=tf.int32)
    ele_emd=[]
    for encoder in all_encoder_models:
        ele_emd.append(encoder([input_ids,attention_mask]))
        encoder.trainable=False
         
    y = Concatenate(axis=1)(ele_emd)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(num_labels,activation='sigmoid')(y)
 
    model = Model([input_ids,attention_mask],[y])
    return model