import numpy as np
import pandas as pd
from numpy import mean
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


BJP = pd.read_csv("../Module 2-Tweet_Data/module2_output/procBJPtweets.csv",names=['tweet'])
INC = pd.read_csv("../Module 2-Tweet_Data/module2_output/procINCtweets.csv",names=['tweet'])

print(BJP.shape)
print(INC.shape)
# print(INC.head(10))
# print(INC.tail(10))
# print(BJP.head(10))
# print(BJP.tail(10))
# print(BJP['tweet'])
# print(INC['tweet'])

analyzer = SentimentIntensityAnalyzer()

def sentiment(tweet):
    tweet = remove_pattern(tweet,"@[\w]+")
    tweet = remove_pattern(tweet,"#[\w]+")
    sentences = sent_tokenize(tweet)
    postemp=[]
    negtemp=[]
    neutemp=[]
    compoundtemp=[]
    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)
        postemp.append(score['pos'])
        negtemp.append(score['neg'])
        neutemp.append(score['neu'])
        compoundtemp.append(score['compound'])
        # print(postemp,negtemp,neutemp,compoundtemp)
    pos = mean(postemp)
    neg = mean(negtemp)
    neu = mean(neutemp)
    compound = mean(compoundtemp)
    if compound >0:
        return "POSITIVE"
    elif compound<0:
        return "NEGATIVE"
    else:
        return "NEUTRAL"



BJP['label'] = BJP.apply(lambda row : sentiment(row['tweet']),axis=1)
INC['label'] = INC.apply(lambda row : sentiment(row['tweet']),axis=1)

BJP = BJP[["label","tweet"]]
INC = INC[["label","tweet"]]
print("**********************")
print("For BJP")
print(BJP["label"].value_counts(normalize=True))
print("for INC")
print(INC["label"].value_counts(normalize=True))

print(BJP.head(10))
print(INC.head(10))
INC.to_csv("./module4_output/INC_Labelled.csv",index=False)
BJP.to_csv("./module4_output/BJP_Labelled.csv",index=False)
