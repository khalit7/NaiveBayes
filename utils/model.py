import numpy as np



class NaiveBayesModel():
    
    def __init__(self,tokenizer,vocab,counter):
        
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.counter = counter
        
        self.p_y = None
        self.p_x_y = None
        
    
    def estimate_parameters_from_data(self,data_itr):
        
        '''
        P(X,y) = P(y) * prod( P(X_i/y) )
        '''
        text="Estimating P(y) and P(X/y) by calculating counts"
        print(f"{text:.^50}")
        
        p_y = np.zeros(2)
        p_x_y = np.zeros((len(self.vocab),2))
        for i,(label,text) in enumerate(data_itr):
            #logging process
            if i%500000 == 0:
                print(f"Finished processing {i} examples")
            
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
        
        return p_y,p_x_y
    
    
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
            
            
        label = pred.argmax()
        
        if label==0:
            return "NEGATIVE"
        elif label==1:
            return "POSITIVE"
        else:
            raise Exception("un recognized label")
            
        
        
        
    