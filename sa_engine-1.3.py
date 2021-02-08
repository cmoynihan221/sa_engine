from __future__ import division
from collections import Counter 
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import math

train_df = pd.read_csv("train.tsv", sep='\t')
test_df = pd.read_csv("dev.tsv", sep='\t')

#Feature selection commented out as no improvement gained
class Classifier:
    ps = PorterStemmer()
    sv = 1
    stop_words = set(stopwords.words('english'))
    
    def __init__(self, df, classes):
        self.classes = classes
        self.datafile = df
        self.tf_sentiment,self.total_words, self.sentiment_totals, self.word_freq, self.total_term, self.word_set = self.count(self.classes)
        #Feature selection, the number of features to select
        #self.feature_select(170)
        print("Total Features: {}".format(len(self.word_set)) )
        #Second count with selected features
        #self.tf_sentiment,self.total_words, self.sentiment_totals, self.word_freq, self.total_term, self.word_set = self.count2(self.classes)
        self.word_probs = self.get_probs()
        
    
    #preprocessing set to stem word and make lowercase
    def preprocess(self, theword):
        word = self.ps.stem(theword.lower())
        return word

    
    #Calculate a term frequence inverse document frequence score for each term
    def tf_idf(self):
        tfidf_values = {}
        classavg = 0
        for term, v in self.word_freq.items():
            tfidf_values[term] = {}
            maxtfidf = 0
            tfidfc = None
            
            for sentiment_class in range(0,self.classes):
                tf = self.tf_sentiment[sentiment_class][term]
                idf = math.log(self.classes/self.word_freq[term])
                tfidf = tf*idf     
                tfidf_values[term][sentiment_class] = tfidf
        return tfidf_values 
                
    
    #Maps sentiment 
    def map_sentiment(self, sentiment):
        if sentiment == 0 or sentiment == 1:
            return 0
        elif sentiment == 2:
            return 1
        elif sentiment == 3 or sentiment == 4:
            return 2     

    #Performs count of terms in sentance and creates data structures
    def count(self,v):
        tf = Counter()
        word_set = set()
        term_sentiment_count = Counter()
        tf_sentiment = {}
        sentiment_totals = Counter()
        total_term = 0
        sentance_totals = {0:0,1:0,2:0,3:0,4:0}
        for i in range(0,v):
            tf_sentiment[i] = Counter()      
        for review in self.datafile.itertuples():
            words = set(review[2].split())
            #map sentiment value
            sentiment = review[3]
            if v == 3:
                sentiment = self.map_sentiment(sentiment)
            sentance_totals[sentiment] +=1
            for word in words:
                #Apply preprocessing
                word = self.preprocess(word)
                if word not in self.stop_words:
                    total_term +=1 
                    if word not in word_set:
                        word_set.add(word)
                    tf_sentiment[sentiment][word] +=1
                    tf[word] +=1
                    sentiment_totals[sentiment]+=1
         
        #total number of unique features
        distinct_features = len(word_set)
        return tf_sentiment, distinct_features, sentiment_totals, tf ,total_term, word_set

    #Performs the same function as the method count() however only counts terms that have been selected by feature selection
    def count2(self,v):
        tf = Counter()
        word_set = set()
        term_sentiment_count = Counter()
        tf_sentiment = {}
        sentiment_totals = Counter()
        total_term = 0 
        for i in range(0,v):
            tf_sentiment[i] = Counter()
        for review in self.datafile.itertuples():
            words = set(review[2].split())
            for word in words:
                word = self.preprocess(word)
                if word not in self.stop_words:
                    if word in self.word_set:
                        if v == 3:
                            sentiment = self.map_sentiment(review[3])
                        else:
                            sentiment = review[3]
                        total_term +=1 
                        
                        if word not in word_set:
                            word_set.add(word)
                        tf_sentiment[sentiment][word] +=1
                        tf[word] +=1
                        sentiment_totals[sentiment]+=1
        distinct_features = len(word_set)
        return tf_sentiment, distinct_features, sentiment_totals, tf ,total_term, word_set

    #returns the probabilities for a term to appear in a sentiment class
    def get_probs(self):
        probs = {}
        for sentiment in range(0,self.classes):
            probs[sentiment] = {}
            class_total = self.sentiment_totals[sentiment]
            for word in self.word_set:
                    if word in self.tf_sentiment[sentiment]:
                        #Adds smoothing value if term is not found in a class
                        probability = self.smooth(self.tf_sentiment[sentiment][word],word,class_total)
                    else:
                        probability = self.smooth(0,word,class_total)
                    probs[sentiment][word] = probability
        return probs
                  
    #Find the class scores for each sentance
    def classify(self, sentance, v):
        match = False
        scores= []
        sentance = sentance.split()
        for i in range(0,v):
            score = 1
            for word in sentance:
                word = self.preprocess(word)
                if word in self.word_probs[i]: #word must be a selected feature
                    #if match remains false then the sentance is unclassifiable
                    if self.word_probs[i][word] > self.smooth(0,word,self.sentiment_totals[i]):
                        match = True
                    score = score*self.word_probs[i][word] 
            scores.append((i,score))
        #sorts the scores to largest first
        return(sorted(scores, key=lambda x: x[1], reverse= True)), match
                    
    
    
    #Laplace smoothing Function           
    def smooth(self,value, term, sent_count):
        smoothed = (value+1)/(sent_count+len(self.word_set))
        return smoothed

    #Iterates through each sentance in a file assigning a sentiment class based on the bayes classifier
    def test_file(self, df):
        mtotal=0
        total = 0
        class_totals = {0:0,1:0,2:0,3:0,4:0, 'na':0}
        f = open("output.txt", 'w')
        #f2 = open("dev_predictions_3classes_Charlie_MOYNIHAN.tsv", 'w')
        #classifies each sentance in the file 
        for review in df.itertuples():
            
            guess,match = self.classify(review[2], self.classes)
            if match == True:
                mtotal +=1
                guessClass = guess[0][0]
            else:
                guessClass = 'na'
            class_totals[guessClass] +=1
            f.write((str)(guessClass)+"\n")
           # f2.write((str)(review[1])+'\t'+str(guessClass)+"\n")
            
            total +=1
        f.close()
        print("Feature Match: {}%".format((mtotal/total)*100))
        print("Class Guess Totals: {}".format(class_totals))

    #Calculate chi score for a term in a sentiment class
    def chiScore(self,term, _class):
        n = self.total_term
        a = self.tf_sentiment[_class][term]     # term frequency of the term in the class
        b = (self.word_freq[term]) - a          # term frequency of the term in all other classes
        c = self.sentiment_totals[_class] - a   # frequency of all other terms in the class
        d = n - self.word_freq[term] - c        # frequency of all other terms in all other classes
        E = ((a*d)-(c*b))**2                    
        chi  = (n*E)/((a+c)*(b+d)*(a+b)*(c+d))
        return chi

    #Feature selection calculates a max chi score for each term, selectes the n highest scoring terms.
    def feature_select(self,n):
        chiScores = []
        best = set()
        totals = {0:0,1:0,2:0,3:0,4:0}
        selected_features = set()
        for word in self.word_set:
            #if word not in self.stop_words:
                maxChi = 0
                curClass = None
                for i in range(0,self.classes):
                    if word in self.tf_sentiment[i]:
                        chiScore = self.chiScore(word,i)
                        if chiScore > maxChi:
                            maxChi = chiScore
                            curClass = i
                if  word not in best:
                    chiScores.append((word,curClass,maxChi))
                    best.add(word)
      
        chiScores = sorted(chiScores, key=lambda x: x[2],reverse=True)[:n]
        for word in chiScores:
            totals[word[1]] +=1
            selected_features.add(word[0])
        self.word_set = selected_features
        print("Feature Class Totals: {}".format(totals))
        print("WORD SET: " +self.word_set)
        
        
        
            
        
        
        
#v number of classes
v= 3
#creates and trains classifier
c = Classifier(train_df, v)
#classifies file
c.test_file(test_df)


#This code is used for feedback on classification of dev set
f=open("output.txt", "r")
correct = 0
total = 0
cf_matrix5 = [[0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]]
cf_matrix3 = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]
                    
if v == 5:
    for sentiment in test_df['Sentiment']:
        guess = f.readline()
        if guess != 'na\n':
            if int(guess) == int(sentiment):
                correct += 1
            total +=1
            cf_matrix5[int(sentiment)][int(guess)] += 1
    cd = sns.heatmap(cf_matrix5, annot=True,fmt='g')
elif v == 3:
    for sentiment in test_df['Sentiment']:
        guess = f.readline()
        sentiment = c.map_sentiment(sentiment)
        if guess != 'na\n':
            if int(guess) == int(sentiment):
                correct += 1
            total +=1
            cf_matrix3[int(sentiment)][int(guess)] += 1
    cd = sns.heatmap(cf_matrix3, annot=True, fmt='g')
            
print("Percentage Correct: {:.2f}".format((correct/total)*100))
plt.ylabel("Class")
plt.xlabel("Guess")
plt.show()
    
