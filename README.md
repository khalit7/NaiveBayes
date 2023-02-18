# Naive Bayes

implementation of a simple Naive Bayes model

The dataset AmazonReviewPolarity was used for training and testing the clasifier. 
basically we learn the distributions P(y) and P(X/y). where X are the features (the words) and y are the labels (negative sentiment or positive sentiment).

## we use the learnt distributions for:

**1. classification:**

  It achieved an accuracy of 85% ( not very naive after all huh? :) )
  
**2. text generation:**

  The idea is to generate words using the distribution P(X/y). since Naive Bayes assumes independance between words (the naive assumption), we should not expect it to generate a meaningfull or cohearent text. However, we should expect it to generate words that signify the label.
  for example if we choose the distribution P(X/y=0) we should expect words that are normally used to negatively talk about a product, on the other hand if we choose the distribution P(X/y=1) we should expect positive words.
  
  ### **Examples of generated outputs:**
  
  from P(X/y=0): 
  
 for terrible of the if the don to if and out time university still charging can army ' mass weren . 1-3 won . is ) what your who . of not hard hate of comes what <unk> trimmers the by back service properly it to this but <unk> problems
 
  from P(X/y=1): 
  
interested and many ' soulful and . write reason second mirror brings when but it there tormented end said directed of always of . be while replace cd-rom . is would entertaining . the sufficed in on ' protect information of and emergency the ! . . episode debut a
