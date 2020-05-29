import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import *

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text
##Display settings in the output
pd.set_option("display.max_colwidth", 100)

#to avoid clutter in the output
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Loading files
train  = pd.read_csv('./train_data/train.csv')
test = pd.read_csv('../Module 2-Tweet_Data/module2_output/procINCtweets.csv',names=['tweet'])

#Removing all rows with non-string values i
train['tweet'] = train['tweet'].apply(lambda x: x if isinstance(x,str) else " ")
test['tweet'] = test['tweet'].apply(lambda x: x if isinstance(x,str) else " ")

#test = pd.read_csv('testt.csv')
print(train.shape)
print(test.shape)
#train.columns= ['id','label','tweet']
print(train[train['label']==0].head(10))
print(train["label"].value_counts())
#test = pd.read_csv('test_tweets.csv')
test["id"] = test.index
test = test[['id','tweet']]
##check the distribution of length of the tweets, in terms of words##
length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
print(length_test,length_train)
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

##Combine the test and train datasets for cleaning##
combined_set = train.append(test,ignore_index=True)
print(combined_set.shape)
# combi['tweet'] = combi['tweet'].apply(lambda x: x if isinstance(x,str) else " ")

#Twitter handles , URLs, punctuation ,numbers ,  special characters , small words,normalize words like loved,loves,loving to love

#handles
combined_set['tidy_tweet'] = np.vectorize(remove_pattern)(combined_set['tweet'],"@[\w]*")
#print(combined_set.head(10))
#everything except letters and hashtags
combined_set['tidy_tweet'] = combined_set['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
#removing short words
combined_set['tidy_tweet'] = combined_set['tidy_tweet'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=3]))
#Stemming the tweets // Normalizing
tokenized_tweet = combined_set['tidy_tweet'].apply(lambda x: x.split())
#print(tokenized_tweet.head(10))
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

#Joining back the tokenized tweets
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

combined_set['tidy_tweet'] = tokenized_tweet
print(combined_set.head(10))

#Outputting to files to be used by other programs
combined_set.to_csv("./module4_output/Combined_preprocessed_tweets.csv",index=False)
ProcessedTrain = combined_set.iloc[:60001,:]
ProcessedTest = combined_set.iloc[60000:,:]
#13933
ProcessedTest.to_csv("./module4_output/ProcessedTest.csv",index=False)
ProcessedTrain.to_csv("./module4_output/ProcessedTrain.csv",index=False)


print("Working!")
