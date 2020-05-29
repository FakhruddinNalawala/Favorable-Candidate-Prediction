import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from tqdm import tqdm
import gensim.downloader as api
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv("./module4_output/ProcessedTrain.csv")
test=pd.read_csv("./module4_output/ProcessedTest.csv")
combi = pd.read_csv("./module4_output/Combined_preprocessed_tweets.csv")
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: x if isinstance(x,str) else " ")

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words="english")
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
print(bow.shape)

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words="english")
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
print(tfidf.shape)

model_w2v = api.load("glove-twitter-25")
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


def word_vector(tokens,size):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count !=0:
        vec /= count
    return vec

word2vec_arrays = np.zeros((len(tokenized_tweet),25))
for i in range(len(tokenized_tweet)):
    word2vec_arrays[i,:] = word_vector(tokenized_tweet[i],25)
    wordvec_df = pd.DataFrame(word2vec_arrays)
print(wordvec_df.shape)

combi = combi.dropna(subset=['label','tweet'])
train = train.dropna(subset=['label','tweet'])

train_bow = bow[:60000,:]
test_bow = bow[60001:,:]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=42,test_size=0.3)


rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain)
prediction = rf.predict(xvalid_bow)
# validation score
print(f1_score(yvalid, prediction_int,average=None))
print(f1_score(yvalid, prediction_int,average='weighted'))
print(f1_score(yvalid, prediction_int,average='micro'))

# test_pred = rf.predict(test_bow)
# test['label'] = test_pred
# submission = test[['id','label']]
# submission.to_csv('sub_rf_bow.csv', index=False)
train_tfidf = tfidf[:60000,:]
test_tfidf = tfidf[60001:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain)
prediction = rf.predict(xvalid_tfidf)
print(f1_score(yvalid, prediction_int,average=None))
print(f1_score(yvalid, prediction_int,average='weighted'))
print(f1_score(yvalid, prediction_int,average='micro'))

train_w2v = wordvec_df.iloc[:60000,:]
test_w2v = wordvec_df.iloc[60001:,:]
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain)
prediction = rf.predict(xvalid_w2v)
print(f1_score(yvalid, prediction_int,average=None))
print(f1_score(yvalid, prediction_int,average='weighted'))
print(f1_score(yvalid, prediction_int,average='micro'))
