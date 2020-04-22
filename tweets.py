import GetOldTweets3 as got
from datetime import datetime

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("BJP,Modi").setSince("2019-01-01").setUntil("2019-04-11").setMaxTweets(50).setNear("12.9716, 77.5946").setWithin("150km");
tweet = got.manager.TweetManager.getTweets(tweetCriteria);

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("INC,Congress,Rahul,Gandhi").setSince("2019-01-01").setUntil("2019-04-11").setMaxTweets(50).setNear("12.9716, 77.5946").setWithin("150km");
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria);

with open("BJPtweets.txt","w") as f:
    for tw in tweet:
        try:
            f.write("Username: "+tw.username+"\n"+"Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Username: "+tw.username+"\n"+"Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass

# id (str)
# permalink (str)
# username (str)
# to (str)
# text (str)
# date (datetime) in UTC
# retweets (int)
# favorites (int)
# mentions (str)
# hashtags (str)
# geo (str)
