import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from tqdm import tqdm
from gensim.models.doc2vec import LabeledSentence
import gensim.downloader as api
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

##Printing info for gensim pre trained word embeddings
# info = api.info()
# print(json.dumps(info, indent=4))
train=pd.read_csv("ProcessedTrain.csv")
test=pd.read_csv("ProcessedTest.csv")
combi = pd.read_csv("Combined_preprocessed_tweets.csv")
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: x if isinstance(x,str) else " ")
# combi = shuffle(combi)
# combi = combi.reset_index(inplace=True,drop=True)
# print(combi.head(10))
# combi = combi.sample(frac=1).reset_index(drop=True)

####
#FEATURE EXTRACTION
####

##BAG_OF_WORDS
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words="english")
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
print(bow.shape)

##TFIDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words="english")
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
print(tfidf.shape)

##to print pre trained word embeddings from gensim-downloader
# for model_name, model_data in sorted(info['models'].items()):
#     print(
#         '%s (%d records): %s' % (
#             model_name,
#             model_data.get('num_records', -1),
#             model_data['description'][:40] + '...',
#         )
#     )
model_w2v = api.load("glove-twitter-25")

##WORD2VEC
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
# model_w2v = gensim.models.Word2Vec(
#             tokenized_tweet,
#             size=200,
#             window=5,
#             min_count=2,
#             sg=1,
#             hs=0,
#             negative=10,
#             workers=2,
#             seed=34)
#
# model_w2v.train(tokenized_tweet, total_examples=len(combi['tidy_tweet']),epochs=20)
#
#print(model_w2v.wv.most_similar(positive="dinner"))
#print(model_w2v.wv.most_similar(positive="trump"))

#print(model_w2v['food'])
#print(len(model_w2v['food']))
#
##VECTORS FOR TWEETS
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
#
#
# #
# tqdm.pandas(desc="progress-bar")
# ##TAGGING EACH TOKENISED TWEET
# def add_label(twt):
#     output = []
#     for i, s in zip(twt.index, twt):
#         output.append(LabeledSentence(s, ["tweet_" + str(i)]))
#     return output
#
# labeled_tweets = add_label(tokenized_tweet) # label all the tweets
# print(labeled_tweets[:6])
#
# ##TRAINNG DOC2VEC
#
# model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
#             dm_mean=1, # dm = 1 for using mean of the context word vectors                                  size=200, # no. of desired features
#             window=5, # width of the context window
#             negative=7, # if > 0 then negative sampling will be used                                 min_count=5, # Ignores all words with total frequency lower than 2.
#             workers=3, # no. of cores
#             alpha=0.1, # learning rate
#             seed = 23)
# model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
# model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)
#
# ##PREPARING DOC2VEC FEATURE SET
# docvec_arrays = np.zeros((len(tokenized_tweet), 200))
# for i in range(len(combi)):
#     docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))
#
# docvec_df = pd.DataFrame(docvec_arrays)
# print(docvec_df.shape)





#MODEL CREATION

# F1 score is being used as the evaluation metric. It is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is suitable for uneven class distribution problems.
#
# The important components of F1 score are:
#
# True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
# True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
# False Positives (FP) – When actual class is no and predicted class is yes.
# False Negatives (FN) – When actual class is yes but predicted class in no.
# Precision = TP/TP+FP
#
# Recall = TP/TP+FN
#
# F1 Score = 2(Recall Precision) / (Recall + Precision)

##Dropping rows with NaN as the value as this was giving lot of errors
combi = combi.dropna(subset=['label','tweet'])
train = train.dropna(subset=['label','tweet'])


##USING BOW FEATURES
train_bow = bow[:60000,:]
test_bow = bow[60001:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=42,test_size=0.3)


lreg = LogisticRegression()
lreg.fit(xtrain_bow,ytrain)  #Traininf the Model
prediction = lreg.predict_proba(xvalid_bow) #Predicting on validation set
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid,prediction_int,average=None))
print(f1_score(yvalid,prediction_int,average='weighted'))
print(f1_score(yvalid,prediction_int,average='micro'))
#
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label','tweet']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

#
# ##USING TFIDF FEATURES
train_tfidf = tfidf[:60000,:]
test_tfidf = tfidf[60001:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int,average=None))
print(f1_score(yvalid, prediction_int,average='weighted'))
print(f1_score(yvalid, prediction_int,average='micro')) # calculating f1 score for the validation set

# ##WORD2VEC FEATURES
train_w2v = wordvec_df.iloc[:60000,:]
test_w2v = wordvec_df.iloc[60001:,:]
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
lreg.fit(xtrain_w2v, ytrain)
prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int,average=None))
print(f1_score(yvalid, prediction_int,average='weighted'    ))
print(f1_score(yvalid, prediction_int,average='micro'))
