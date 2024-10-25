
# In[1]:


from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[2]:
import nltk
nltk.download('punkt')
nltk.download('wordnet')

data = open('./project2_training_data.txt', encoding="utf8")


# In[3]:


def Convert(string):
    li = list(string.split(" "))
    return li


# In[4]:


def nostopword(doc):
    filtered_sentence = []
    for i in doc:
        filtered = remove_stopwords(i)
        filtered_sentence.append(filtered)
    return filtered_sentence


# In[5]:


text = data.read()
print("Tokenizing sentences from training data:\n")
print(sent_tokenize(text)[0:5])
print("\nTokenizing words from training data:\n")
print(word_tokenize(text)[0:10])
print("\nTotal number of sentences in data:\n")
print(len(sent_tokenize(text)))


# In[6]:


data = open('./project2_training_data.txt', encoding="utf8")
text = data.read().split('\n')
train = []
for t in text:
    train.append([t])


# In[7]:


import csv,sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest,chi2 
from nltk.stem import PorterStemmer

# Load data
fl=open('./project2_training_data.txt',encoding="utf8")  # Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
reader = list(csv.reader(fl,delimiter='\n'))
f2 = open('./project2_training_data_labels.txt',encoding="utf8")
reader2 = list(csv.reader(f2,delimiter='\n'))
# Load test data
f3=open('./project2_test_data.txt',encoding="utf8")  # Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
reader3 = list(csv.reader(f3,delimiter='\n'))


test = []
for i in reader3:
    test.append(i[0])
data=[]; labels=[];
for i,item in enumerate(reader2):
    if item[0]=='positive':
        labels.append(0)
    elif item[0] == 'negative':
        labels.append(1)
    else:
        labels.append(2)
    data.append(reader[i][0])


# In[8]:


def Classification(data,labels,test=None):
    opt1=input('Enter\n\t "a" for Simple Classification \n\t "b" for Classification with Cross Validation \n\t "q" to quit \n')
    opt3=input('Enter\n\t "w" for with stop_words \n\t "wo" for without stopwords \n\t "q" to quit \n')
    
    if opt1=='a':            # simple run with no parameter tuning
        if opt3== 'w':
            stop_words = stopwords.words('english')
            vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
        elif opt3== 'wo':
            vectorizer=TfidfVectorizer(ngram_range=(1,3),token_pattern=r'\b\w+\b')
        else:
            sys.exit(0)  
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)
    #    clf = RandomForestClassifier(criterion='gini',class_weight='balanced') 

        
        tfidf = vectorizer.fit_transform(data)
        terms=vectorizer.get_feature_names()
        tfidf = tfidf.toarray()
        # Training and Test Split           
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(tfidf, labels, test_size=0.20, random_state=42,stratify=labels)   

        #Classificaion    
        clf.fit(trn_data,trn_cat)
        predicted = clf.predict(tst_data)
        predicted =list(predicted)

    elif opt1=='b':          # parameter tuning using grid search
        # Training and Test Split
        if opt3=='w':
            data = data
            test = test
        elif opt3 =='wo':
            data = nostopword(data)
            test1 = []
            for i in reader3:
                test1.append(i[0])
            test = nostopword(test1)
        else:
            sys.exit(0)
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
        opt2 = input("Choose a classifier : "
                       "\n\t 's' to select Linear SVC" 
                       "\n\t 'ls' to select SVM" 
                       "\n\t 'dt' to select Decision Tree"   
                       "\n\t 'rf' to select Random Forest"
                       "\n\t 'mn' to select multinomial naive bayes \n\n")    
    # Naive Bayes Classifier    
        if opt2=='mn':      
            clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1),
            }  
    # SVM Classifier
        elif opt2=='ls': 
            clf = svm.LinearSVC(class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.2,0,3,0.4,0.5,0.6,0.7,0.8,0.9,1),
            }   
        elif opt2=='s':
            clf = svm.SVC(kernel='linear', class_weight='balanced')  
            clf_parameters = {
            'clf__C':(0.1,0.5,1,2,10,50,100),
            }   
    # Decision Tree Classifier
        elif opt2=='dt':
            clf = DecisionTreeClassifier(random_state=40)
            clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':('sqrt', 'log2'),
            'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
            }  
    # Random Forest Classifier    
        elif opt2=='rf':
            clf = RandomForestClassifier(class_weight='balanced', max_depth=10)
            clf_parameters = {
                        'clf__criterion':('gini', 'entropy'), 
                        'clf__max_features':('sqrt', 'log2'),   
                        'clf__n_estimators':(30,50,100,200),
                        'clf__max_depth':(10,20),
                        }     
        else:
            print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
            sys.exit(0)                                  
    # Feature Extraction
        pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3))),

        ('feature_selector', SelectKBest(chi2, k=600)),         
        ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
        ('clf', clf),]) 

        feature_parameters = {
        'vect__min_df': (2,3),
        'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
        }

    # Classificaion
        parameters={**feature_parameters,**clf_parameters} 
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_  
        print('********* Best Set of Parameters ********* \n\n')
        print(clf)

        predicted = clf.predict(tst_data)
        predicted =list(predicted)
        if test!= None:
            predicted_test = list(clf.predict(test))
            
    else:
        sys.exit(0)

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
    print ('\n Confusion Matrix \n')  
    print (confusion_matrix(tst_cat, predicted))  
    print(classification_report(tst_cat,predicted))
    if test==None:
        return clf
    return clf, predicted_test
    # pr=precision_score(tst_cat, predicted, average='binary') 
    # print ('\n Precision:'+str(pr)) 

    # rl=recall_score(tst_cat, predicted, average='binary') 
    # print ('\n Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='binary') 
    #print ('\n Micro Averaged F1-Score:'+str(fm))


# In[9]:


def bestclassifi(data,labels,test=None):
    
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   
  
    # Naive Bayes Classifier    

    # SVM Classifier 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.2,0,3,0.4,0.5,0.6,0.7,0.8,0.9,1),
        }   
                              
    # Feature Extraction
        pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3))),

        ('feature_selector', SelectKBest(chi2, k=600)),         
        ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
        ('clf', clf),]) 

        feature_parameters = {
        'vect__min_df': (2,3),
        'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
        }

    # Classificaion
        parameters={**feature_parameters,**clf_parameters} 
        grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
        grid.fit(trn_data,trn_cat)     
        clf= grid.best_estimator_  
        print('********* Best Set of Parameters ********* \n\n')
        print(clf)

        predicted = clf.predict(tst_data)
        predicted =list(predicted)
        if test!= None:
            predicted_test = list(clf.predict(test))
            

    # Evaluation
        print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
        print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
        print ('\n Confusion Matrix \n')  
        print (confusion_matrix(tst_cat, predicted))  
        print(classification_report(tst_cat,predicted))
        if test==None:
            return clf
        return clf, predicted_test
    # pr=precision_score(tst_cat, predicted, average='binary') 
    # print ('\n Precision:'+str(pr)) 

    # rl=recall_score(tst_cat, predicted, average='binary') 
    # print ('\n Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='binary') 
    #print ('\n Micro Averaged F1-Score:'+str(fm))


# In[10]:


def show(predicted):
    R = []
    for i in range(0,len(predicted)-1):
        if predicted[i] == 0:
            R.append("positive")
        elif predicted[i] == 1:
            R.append("negative")
        elif predicted[i] == 2:
            R.append("neutral")
    return R


# In[11]:


def stemmed(doc):
    doc1 = []
    
    for i in doc:
        words=""
        for j in word_tokenize(i):
            words +=porter.stem(j)
            words+=' '
        doc1.append(words)
    
    return doc1


# In[12]:


def leme(doc):
    doc1 = []
    
    for i in doc:
        words=""
        for j in word_tokenize(i):
            words +=lemmatizer.lemmatize(j)
            words+=' '
        doc1.append(words)
    
    return doc1


# In[13]:


print("\nData without stemming and lemmatization\n")
print(data[0:5])
print("\nData with stemming\n")
print(stemmed(data)[0:5])
print("\nData with lemmatization\n")
print(leme(data)[0:5])


# In[14]:


print("\n\t PLEASE WAIT, TRAINING THE BEST MODEL AND SAVING IT'S PREDICTIONS AS CSV FILE....\n")


# In[15]:


a = stemmed(data)
b = stemmed(test)
clf = bestclassifi(a,labels)
pred = clf.predict(b)
print(show(pred))


# In[18]:


import numpy as np
import pandas as pd
prediction = pd.DataFrame(show(pred), columns=['predictions']).to_csv('./prediction.csv', encoding="utf8", header=False, index=False)


# In[19]:


print("\n\t Now here we can try any model we want, by giving required inputs (as many times as we like).")


# In[ ]:


while True:
    inp=input('Enter\n\t "st" for Stemming \n\t "l" for lemmatization \n\t "s"for none: \n')
    if inp =="st":
        data = stemmed(data)
        test = stemmed(test)
        clf = Classification(data,labels)
        pred = clf.predict(test)
        print(show(pred))
    elif inp=="l":
        data = leme(data)
        test = leme(test)
        clf = Classification(data,labels)
        pred = clf.predict(test)
        print(show(pred))
    elif inp=="s":
        data = data
        test = test
        clf = Classification(data,labels)
        pred = clf.predict(test)
        print(show(pred))
    else:
        sys.exit(0)

