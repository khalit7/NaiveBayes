import torch
import os
import numpy as np
from sklearn.metrics import classification_report
from random import choices

import dataset


class NaiveBayesModel():
    
    def __init__(self):
        
        self.tokenizer = None
        self.vocab = None
        
        self.p_y = None
        self.p_x_y = None
        
    def __call__(self,text):
        pred_label = self.get_prediction_from_text(text)
        
        if pred_label == 1:
            return "POSITIVE"
        elif pred_label == 0:
            return "NEGATIVE"
        
    def get_vocab_and_tokenizer(self,data_itr):
        tokenizer = dataset._get_tokenizer()
        print("building vocabulary from dataset ... ",end=" ")
        vocab = dataset.build_vocab(data_itr,tokenizer)
        print("Done!")
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    
    def estimate_parameters_from_data(self,data_itr):
        
        '''
        P(X,y) = P(y) * prod( P(X_i/y) )
        '''
        text="Estimating P(y) and P(X/y) by calculating counts"
        print(f"{text:.^100}")
        
        p_y = np.zeros(2)
        p_x_y = np.zeros((len(self.vocab),2))
        for i,(label,text) in enumerate(data_itr,1):
            #logging process
            if i%500000 == 0:
                print(f"***Finished processing {i} examples")
            
            p_y[label] +=1  # calculate P(y) -the bias term-  ==> this is simply the proportion of a specific label to the number of examples in data
            for token in self.tokenizer(text):
                p_x_y[self.vocab[token]][label] +=1 # calculate P(X/y) ===> this will be a VOCAB_SIZE * NUM_OF_LABELS array where the element i,j represents the probability of x_i given label y_j

        # normalize P(y) to make it a probability distribution
        n = sum(p_y)
        p_y[0] = p_y[0]/n
        p_y[1] = p_y[1]/n


        # normalize P(X/y) to make it a probability distribution .. using add-1 smoothing
        p_x_y = (p_x_y+1)/(p_x_y.sum(axis=0)+len(self.vocab))
        
        self.p_y = p_y
        self.p_x_y = p_x_y
        
        text="Done!"
        print(f"{text:.^100}")
        return p_y,p_x_y
    
    def save_parameters(self,path):
        p_y_path = os.path.join(path,"p_y.npy")
        np.save(p_y_path,self.p_y)
        
        p_x_y_path = os.path.join(path,"p_x_y.npy")
        np.save(p_x_y_path, self.p_x_y)
        
        vocab_path = os.path.join(path,'vocab.pth')
        torch.save(self.vocab, vocab_path)
        
    def load_parameters(self,path):
        p_y_path = os.path.join(path,"p_y.npy")
        self.p_y = np.load(p_y_path)
        
        p_x_y_path = os.path.join(path,"p_x_y.npy")
        self.p_x_y = np.load(p_x_y_path)
        
        # vocab
        vocab_path = os.path.join(path,'vocab.pth')
        self.vocab =  torch.load(vocab_path)
        # tokenizer
        self.tokenizer = dataset._get_tokenizer()
        
    
    
    def get_prediction_from_text(self,text):
        
        text_tokens = self.tokenizer(text)
        
        pred = np.zeros(2)
        
        # add the bias term
        pred[0] += np.log(self.p_y[0])
        pred[1] += np.log(self.p_y[1])

        # add P(X/y)
        for token in text_tokens:
            # calculate prop that it belongs to label 0
            label_0_prob = self.p_x_y[self.vocab[token]][0]
            pred[0] += np.log(label_0_prob)
            # calculate prop that it belongs to label 1
            label_1_prob = self.p_x_y[self.vocab[token]][1]
            pred[1] += np.log(label_1_prob)
            
            
        predicted_label = pred.argmax()
        
        return predicted_label
            
    def calculate_performance_on_data(self,data):

        y_true = []
        y_pred = []
        
        for i,(label,text) in enumerate(data):
            #logging process
            if i%500000 == 0:
                print(f"***Finished processing {i} examples")
                
            y_true.append(label)
            
            pred = self.get_prediction_from_text(text)
            y_pred.append(pred)
            
        
        return classification_report(y_true,y_pred,output_dict=True)
        
    
    def generate_text(self,label,num_words=50):
        
        distribution = self.p_x_y[:,label].tolist()
        
        words = self.vocab.lookup_tokens(range(len(self.vocab)))
        
        generated_text = " ".join(choices(words,distribution,k=num_words))
        
        return generated_text
            