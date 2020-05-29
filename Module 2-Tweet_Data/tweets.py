import GetOldTweets3 as got
from datetime import datetime

with open("BJPtweets.txt","w") as f:
    pass
with open("INCtweets.txt","w") as f:
    pass

#January:

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("BJP").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("Modi").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("PMOIndia").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("INC").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","a") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("SoniaGandhi").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-01-01").setUntil("2019-02-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

#February:

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("BJP").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("Modi").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("PMOIndia").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("INC").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","a") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("SoniaGandhi").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-02-01").setUntil("2019-03-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

#March:

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("BJP").setSince("2019-03-01").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("Modi").setSince("2019-03-01").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("PMOIndia").setSince("2019-03-01").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("INC").setSince("2019-03-01").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","a") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("SoniaGandhi").setSince("2019-03-01").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-03-01").setUntil("2019-03-12").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-03-12").setUntil("2019-03-22").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-03-22").setUntil("2019-04-01").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

#April:

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("BJP").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("Modi").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("PMOIndia").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)

with open("BJPtweets.txt","a") as f:
    for tw in tweet:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("INC").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","a") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("SoniaGandhi").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID :"+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")
        
tweetCriteria = got.manager.TweetCriteria().setQuerySearch("RahulGandhi").setSince("2019-04-01").setUntil("2019-04-11").setNear("12.9716, 77.5946").setWithin("3000km")
tweet1 = got.manager.TweetManager.getTweets(tweetCriteria)

with open("INCtweets.txt","w") as f:
    for tw in tweet1:
        try:
            f.write("Date: "+tw.date.strftime("%c")+"\nID: "+tw.id+"\n")
            f.write(tw.text+"\n\n")
        except:
            pass
print("Done")

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
