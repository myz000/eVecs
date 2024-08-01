import json
import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Model,load_model,Sequential
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input,Concatenate,Lambda,Dot,Activation
from tensorflow.keras.metrics import AUC

class Trainer():
    def __init__(self,
                 element_train_filepath,
                 element_test_filepath,
                 aspects,
                 saved_filepath,
                 train_filepath,
                 test_filepath,
                 classifier_path,
                 epoch,
                 batch_size,
                 encoder_name,
                 classifier_name,
                 tokenizer,
                 MAX_SEQUENCE_LENGTH,
                 bertModel,
                 Id2Data_train,
                 ):
        self.element_train_filepath=element_train_filepath
        self.element_test_filepath=element_test_filepath
        self.aspects=aspects
        self.saved_filepath=saved_filepath
        self.train_filepath=train_filepath
        self.test_filepath=test_filepath
        self.classifier_path=classifier_path
        self.nb_epoch=epoch
        self.batch_size=batch_size
        self.encoder_name=encoder_name
        self.classifier_name=classifier_name
        self.tokenizer = tokenizer
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.bertModel = bertModel
        self.Id2Data_train = Id2Data_train
        self.element_train_datas=self.get_element_data(self.element_train_filepath)
        self.element_test_datas=self.get_element_data(self.element_test_filepath)
       
    def get_train_classifier_data(self):
        train_sen,train_labels = pre_classifier_data(self.train_filepath)
        test_sen,test_labels = pre_classifier_data(self.test_filepath)
        
        train_data = get_bert_input_data(train_sen, train_labels, {},self.MAX_SEQUENCE_LENGTH, self.tokenizer)
        test_data = get_bert_input_data(test_sen, test_labels, {},self.MAX_SEQUENCE_LENGTH, self.tokenizer)
        
        return train_data, test_data
        
    def get_element_data(self,file_path):
        element_train_datas={}
        file_names=os.listdir(file_path)
        for name in file_names:
            with open(file_path+name,'r',encoding='utf8')as fp:
                k=name.split('.')[0]
                element_train_datas[k]=json.load(fp) 
        return element_train_datas

    def get_element_triplets(self, current_k):
            train=self.element_train_datas[current_k]
            test = self.element_test_datas[current_k]
            np.random.shuffle(train)
            s_data=np.zeros((len(train),1))
            d_data=np.zeros((len(train),1))
            o_data=np.zeros((len(train),1))
            y_data=np.zeros((len(train),1))
            for i in range(0,len(train)//2): 
                d=train[i]
                s_data[i] = d['s']
                d_data[i] = d['d']
                o_data[i] = d['o']
                y_data[i] = 1
            for i in range(len(train)//2,len(train)): 
                d=train[i]
                s_data[i] = d['o']
                d_data[i] = d['d']
                o_data[i] = d['s']
                y_data[i] = 0

            indices=np.arange(len(train))
            np.random.shuffle(indices)
            s_data=s_data[indices]
            d_data=d_data[indices]
            o_data=o_data[indices]
            y_data=y_data[indices]
            
            s_bert_input = get_bert_input_data(s_data, np.zeros((len(s_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
            d_bert_input = get_bert_input_data(d_data, np.zeros((len(d_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
            o_bert_input = get_bert_input_data(o_data, np.zeros((len(o_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
           
            s_test_data=np.zeros((len(test),1))
            d_test_data=np.zeros((len(test),1))
            o_test_data=np.zeros((len(test),1))
            y_test_data=np.zeros((len(test),1))
            
            for i in range(0,len(test)//2): 
                d=test[i]
                s_test_data[i] = d['s']
                d_test_data[i] = d['d']
                o_test_data[i] = d['o']
                y_test_data[i]=1
            for i in range(len(test)//2,len(test)): 
                d=test[i]
                s_test_data[i] = d['o']
                d_test_data[i] = d['d']
                o_test_data[i] = d['s']
                y_test_data[i]=0
        
            
            s_bert_test_input = get_bert_input_data(s_test_data, np.zeros((len(s_test_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
            d_bert_test_input = get_bert_input_data(d_test_data, np.zeros((len(d_test_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
            o_bert_test_input = get_bert_input_data(o_test_data, np.zeros((len(o_test_data))), self.Id2Data_train, self.MAX_SEQUENCE_LENGTH, self.tokenizer)[0]
            
            
            
            train_inputs = [s_bert_input[0],d_bert_input[0],o_bert_input[0],
                            s_bert_input[1],d_bert_input[1],o_bert_input[1]]
            test_inputs = [s_bert_test_input[0],d_bert_test_input[0],o_bert_test_input[0],
                          s_bert_test_input[1],d_bert_test_input[1],o_bert_test_input[1]]
            
            return train_inputs, y_data, test_inputs, y_test_data
    
    def encoder_trainer(self):
        all_history=[]
        for current_k in self.aspects:
            train_inputs, y_data, test_inputs, y_test_data = self.get_element_triplets(current_k)

            element_predict_model=Create_Encoder_Bert_model(self.bertModel, self.MAX_SEQUENCE_LENGTH)
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            element_predict_model.compile(loss=max_margin_loss,optimizer=optimizer,metrics = [my_accuracy])

            history=element_predict_model.fit(x=train_inputs,
                                              y=[y_data],
                                              batch_size=self.batch_size,
                                              epochs=2,
                                              validation_data=(test_inputs,[y_test_data])
                                             )

            all_history.append(history)
            filepath=self.saved_filepath+self.encoder_name+'_encoder/'+current_k+'_bert_encoder_model'
            element_predict_model.save(filepath)
            
        return all_history
    
    def get_encoders_models(self):
        element_train_datas={}
        file_names=os.listdir(self.saved_filepath+self.encoder_name+'_encoder/')
        all_models=[]
        for name in self.aspects:
            aspect_bert_dir = self.saved_filepath+self.encoder_name+'_encoder/'+name+'_bert_encoder_model'
            aspect_model=load_model(aspect_bert_dir,custom_objects={'max_margin_loss':max_margin_loss,'my_accuracy':my_accuracy})
            aspect_predict_model=Model(inputs=[aspect_model.inputs[0],aspect_model.inputs[3]],
                           outputs=[aspect_model.get_layer('output_s').output])
            all_models.append(aspect_predict_model)
        return all_models
    
    def multilabel_classifier(self,all_encoder_models, train_data, test_data):
            
        filepath=self.classifier_path+classifier_name+'/CNN_classifier.tf'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',  save_best_only=True,
                                    mode='min')

        classifier=aspects_CNN_classifier(128,10,all_encoder_models,self.MAX_SEQUENCE_LENGTH)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        classifier.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy',AUC()])

        classifier.fit(train_data[0], train_data[1],
                  batch_size=self.batch_size,
                  epochs=10,
                  validation_data=(test_data[0], test_data[1]),
                  callbacks =[checkpoint])
    
    def classifier_train(self):
        all_encoder_models=self.get_encoders_models()
        train_data, test_data=self.get_train_classifier_data()
        self.multilabel_classifier(all_encoder_models,train_data, test_data)
        
        
class Tester():
    
    def __init__(self,aspects,saved_filepath,test_filepath,classifier_path,
                 encoder_name,classifier_name,
                 tokenizer,MAX_SEQUENCE_LENGTH):
        self.aspects=aspects
        self.saved_filepath=saved_filepath
        self.test_filepath=test_filepath
        self.classifier_path=classifier_path
        self.encoder_name = encoder_name
        self.classifier_name = classifier_name
        self.tokenizer = tokenizer
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
       
    def get_test_classifier_data(self):
        test_datas = pd.read_csv(self.test_filepath)
        test_sen,test_labels = pre_classifier_data(test_texts)
        test_data = get_bert_input_data(test_sen, test_labels, {},self.MAX_SEQUENCE_LENGTH, self.tokenizer)
        
        return test_data[0], test_data[1]
   
    def get_classifier_models(self):
        classifier_model = load_model(self.classifier_path+classifier_name+'/CNN_classifier.tf')
        return classifier_model
   
    
    def find_optim_threshold(self,test_out,test_labels):
        y_pred =  np.array([[1 if test_out[i,j]>=0.5 else 0 for j in range(test_labels.shape[1])] for i in range(len(test_labels))])
        return y_pred
    
    def ele_evaluate(self,test_labels,y_pred):
        label_nums=len(self.aspects)
        result_count=np.zeros((label_nums,4))
        for i in range(len(test_labels)):
            for j in range(test_labels.shape[1]):
                if(test_labels[i,j]==1 and y_pred[i,j]==1):
                    result_count[j,0]+=1 #TP
                if(test_labels[i,j]==0 and y_pred[i,j]==1):
                    result_count[j,1]+=1 #FP
                if(test_labels[i,j]==1 and y_pred[i,j]==0):
                    result_count[j,2]+=1 #FN
                if(test_labels[i,j]==0 and y_pred[i,j]==0):
                    result_count[j,3]+=1 #TN

        precsion=0
        recall=0
        f_macro=0
        for i in range(len(result_count)):
            if result_count[i,0]==0 and result_count[i,1]==0 and result_count[i,2]==0:
                p=r=f=0
            elif result_count[i,0]==0 and (result_count[i,1]>0 or result_count[i,2]>0):
                p=r=f=0
            else:      
                p=result_count[i,0]/(result_count[i,0]+result_count[i,1])
                r=result_count[i,0]/(result_count[i,0]+result_count[i,2])
                f=2*p*r/(p+r)

            precsion+=p
            recall+=r
            f_macro+=f


        precsion=precsion/10
        recall=recall/10
        f_macro=f_macro/10


        TP=np.sum(result_count[:,0])
        FP=np.sum(result_count[:,1])
        FN=np.sum(result_count[:,2])
        TN=np.sum(result_count[:,3])


        pp=TP/(TP+FP)
        rr=TP/(TP+FN)
        f_micro=2*pp*rr/(pp+rr)

        for i in range(test_labels.shape[1]):
            TP=np.sum(result_count[i,0])
            FP=np.sum(result_count[i,1])
            FN=np.sum(result_count[i,2])
            TN=np.sum(result_count[i,3])
            ppp=TP/(TP+FP)
            rrr=TP/(TP+FN)
            f_micro_1=2*ppp*rrr/(ppp+rrr)
            
        return precsion,recall,f_macro,pp,rr,f_micro,(f_macro+f_micro)/2 

        
    def test(self):
        classifier_models=self.get_classifier_models()
        test_tokenizer,test_labels=self.get_test_classifier_data()
        test_out=np.zeros((len(test_labels),len(self.aspects)))
        test_out=classifier_models.predict(test_tokenizer)     
        y_pred=self.find_optim_threshold(test_out,test_labels)
        self.ele_evaluate(test_labels,y_pred)
        
    def get_test_pred(self):
        classifier_models=self.get_classifier_models()
        test_tokenizer,test_labels=self.get_test_classifier_data()
        test_out=np.zeros((len(test_labels),len(self.aspects)))
        test_out=classifier_models.predict(test_tokenizer)
        return test_out, test_labels
   
    def get_results(self):
        classifier_models=self.get_classifier_models()
        test_tokenizer,test_labels=self.get_test_classifier_data()        
        element_test_tokenizer=np.zeros((len(test_tokenizer),len(self.aspects),128))      
        test_out=np.zeros((len(test_labels),len(self.aspects)))
        test_out=classifier_models.predict(test_tokenizer)             
        y_pred=self.find_optim_threshold(test_out,test_labels)
        return self.ele_evaluate(test_labels,y_pred)