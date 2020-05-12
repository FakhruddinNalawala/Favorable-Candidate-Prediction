import re
import nltk
from nltk.stem.porter import *

import numpy as np
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text

INC_df = pd.read_csv("INC_Labelled.csv")
BJP_df = pd.read_csv("BJP_Labelled.csv")
print(BJP_df.head(10))
print(INC_df.head(10))

#Removing all rows with non-string values i
BJP_df['tweet'] = BJP_df['tweet'].apply(lambda x: x if isinstance(x,str) else " ")
INC_df['tweet'] = INC_df['tweet'].apply(lambda x: x if isinstance(x,str) else " ")

BJP_df['tidy_tweet'] = np.vectorize(remove_pattern)(BJP_df['tweet'],"#[\w]*")
INC_df['tidy_tweet'] = np.vectorize(remove_pattern)(INC_df['tweet'],"#[\w]*")


#everything except letters and hashtags
BJP_df['tidy_tweet'] = BJP_df['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
INC_df['tidy_tweet'] = INC_df['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")

##keeping words with len>=3
BJP_df['tidy_tweet'] = BJP_df['tidy_tweet'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=3]))
INC_df['tidy_tweet'] = INC_df['tidy_tweet'].apply(lambda x: ' '.join([word for word in x.split() if len(word)>=3]))

#Stemming the tweets // Normalizing
tokenized_tweet_BJP = BJP_df['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet_INC = INC_df['tidy_tweet'].apply(lambda x: x.split())
#print(tokenized_tweet.head(10))
stemmer = PorterStemmer()
tokenized_tweet_BJP = tokenized_tweet_BJP.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet_INC = tokenized_tweet_INC.apply(lambda x: [stemmer.stem(i) for i in x])

#Joining back the tokenized tweets
for i in range(len(tokenized_tweet_BJP)):
    tokenized_tweet_BJP[i] = " ".join(tokenized_tweet_BJP[i])
for i in range(len(tokenized_tweet_INC)):
    tokenized_tweet_INC[i] = " ".join(tokenized_tweet_INC[i])

BJP_df['tidy_tweet'] = tokenized_tweet_BJP
INC_df['tidy_tweet'] = tokenized_tweet_INC
print(BJP_df.head(10))
print(INC_df.head(10))

##OVERALL VISUALIZATION

#BJP VISUALIZATIONS
#WORDCLOUD for all words
all_words = ' '.join([text for text in BJP_df['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("ALL WORDS BJP")
plt.savefig('images/wordcloud_all_BJP.png')
plt.show()
#
# #WORDCLOUD for positive tweets
pos_words = ' '.join([text for text in BJP_df['tidy_tweet'][BJP_df['label']=="POSITIVE"]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(pos_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("POSITIVE BJP")
plt.savefig('images/wordcloud_pos_BJP.png')
plt.show()
#
# #WORDCLOUD for negative tweets
neg_words = ' '.join([text for text in BJP_df['tidy_tweet'][BJP_df['label']=="NEGATIVE"]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(neg_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("NEGATIVE BJP")
plt.savefig('images/wordcloud_neg_BJP.png')
plt.show()


##INC VISUALIZATIONS
#WORDCLOUD for all words
all_words = ' '.join([text for text in INC_df['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("ALL WORDS INC")
plt.savefig('images/wordcloud_all_INC.png')
plt.show()
#
# #WORDCLOUD for positive tweets
pos_words = ' '.join([text for text in INC_df['tidy_tweet'][INC_df['label']=="POSITIVE"]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(pos_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("POSITVE INC")
plt.savefig('images/wordcloud_pos_INC.png')
plt.show()
#
# #WORDCLOUD for negative tweets
neg_words = ' '.join([text for text in INC_df['tidy_tweet'][INC_df['label']=="NEGATIVE"]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(neg_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Negative INC")
plt.savefig('images/wordcloud_neg_INC.png')
plt.show()


##FOR HASHTAGS
def extract_hashtag(x):
    hashtags = []
    #print(type(x))
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

##extracting hastags from positive tweets
pos_hashtags_BJP = extract_hashtag(BJP_df['tweet'][BJP_df['label'] == "POSITIVE"])
##extracting hastags from negative tweets
neg_hashtags_BJP = extract_hashtag(BJP_df['tweet'][BJP_df['label']=="NEGATIVE"])

pos_hashtags_INC = extract_hashtag(INC_df['tweet'][INC_df['label'] == "POSITIVE"])
##extracting hastags from negative tweets
neg_hashtags_INC = extract_hashtag(INC_df['tweet'][INC_df['label']=="NEGATIVE"])
#unnesting list
pos_hashtags_BJP = sum(pos_hashtags_BJP,[])
neg_hashtags_BJP = sum(neg_hashtags_BJP,[])

pos_hashtags_INC = sum(pos_hashtags_INC,[])
neg_hashtags_INC = sum(neg_hashtags_INC,[])

##BJP 

##BARPLOT FOR pos_hashtags
a = nltk.FreqDist(pos_hashtags_BJP)
d1 = pd.DataFrame({"Hashtag":list(a.keys()),"Count": list(a.values())})
d1 = d1.nlargest(columns="Count",n=20)
plt.figure(figsize=(14,5))
ax1 = sns.barplot(data=d1, x="Hashtag", y="Count")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.set(ylabel="Count")
plt.title("BJP POSITIVE HASHTAGS")
plt.savefig('images/pos_hashtags_BJP.png')
plt.show()

##BARPLOT FOR neg_hashtags
b = nltk.FreqDist(neg_hashtags_BJP)
d2 = pd.DataFrame({"Hashtag":list(b.keys()),"Count": list(b.values())})
d2 = d2.nlargest(columns="Count",n=20)
plt.figure(figsize=(14,5))
ax2 = sns.barplot(data=d2, x="Hashtag", y="Count")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax2.set(ylabel="Count")
plt.title("BJP NEGATIVE HASHTAGS")
plt.savefig('images/neg_hashtags_BJP.png')
plt.show()


##INC
##BARPLOT FOR pos_hashtags
a = nltk.FreqDist(pos_hashtags_INC)
d1 = pd.DataFrame({"Hashtag":list(a.keys()),"Count": list(a.values())})
d1 = d1.nlargest(columns="Count",n=20)
plt.figure(figsize=(14,5))
ax1 = sns.barplot(data=d1, x="Hashtag", y="Count")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax1.set(ylabel="Count")
plt.title("INC POSITIVE HASHTAGS")
plt.savefig('images/pos_hashtags_INC.png')
plt.show()

##BARPLOT FOR neg_hashtags
b = nltk.FreqDist(neg_hashtags_INC)
d2 = pd.DataFrame({"Hashtag":list(b.keys()),"Count": list(b.values())})
d2 = d2.nlargest(columns="Count",n=20)
plt.figure(figsize=(14,5))
ax2 = sns.barplot(data=d2, x="Hashtag", y="Count")
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
)
ax2.set(ylabel="Count")
plt.title("INC NEGATIVE HASHTAGS")
plt.savefig('images/neg_hashtags_INC.png')
plt.show()
