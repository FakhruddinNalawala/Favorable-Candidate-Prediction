import re
import nltk
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
print("Working!")

combi = pd.read_csv("../Module 4-Sentiment_Analysis/module4_output/Combined_preprocessed_tweets.csv")
print(combi.shape)
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: x if isinstance(x,str) else " ")


#WORDCLOUD for all words
all_words = ' '.join([text for text in BJP_df['tidy_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#
# #WORDCLOUD for positive tweets
pos_words = ' '.join([text for text in BJP_df['tidy_tweet'][combi['label']==4]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(pos_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#
# #WORDCLOUD for negative tweets
neg_words = ' '.join([text for text in BJP_df['tidy_tweet'][combi['label']==0]])
wordcloud = WordCloud(width=800, height=500, random_state=21,max_font_size=110).generate(neg_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


##FOR HASHTAGS
def extract_hashtag(x):
    hashtags = []
    #print(type(x))
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags
#
##extracting hastags from positive tweets
# pos_tweets = combi[combi['label']==4]
# pos_list = list(pos_tweets['tidy_tweet'])
# print(type(pos_list))
# pos_hashtags = []
# for i in pos_list:
#     pos_hashtags.extend(extract_hashtag(i))
# #pos_hashtags = extract_hashtag(pos_list)
#
#print(pos_hashtags)
pos_hashtags = extract_hashtag(combi['tidy_tweet'][combi['label'] == 4])
##extracting hastags from negative tweets
neg_hashtags = extract_hashtag(combi['tidy_tweet'][combi['label']==0])
#unnesting list
pos_hashtags = sum(pos_hashtags,[])
neg_hashtags = sum(neg_hashtags,[])

##BARPLOT FOR pos_hashtags
a = nltk.FreqDist(pos_hashtags)
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
plt.show()

##BARPLOT FOR neg_hashtags
b = nltk.FreqDist(neg_hashtags)
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
plt.show()
