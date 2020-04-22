# Election
Okay so this will be the working directory for our project. 

TWEETS.PY  </br>
This is the script that will scrape tweets from twitter .   
Set up a mongodb database and query the results into the database .   
We cant set all these fields into the querysearch using this Library . Use as Required .    

setUsername (str or iterable): An optional specific username(s) from a twitter account (with or without "@").  
setSince (str. "yyyy-mm-dd"): A lower bound date (UTC) to restrict search.  
setUntil (str. "yyyy-mm-dd"): An upper bound date (not included) to restrict search.  
setQuerySearch (str): A query text to be matched.
setTopTweets (bool): If True only the Top Tweets will be retrieved.
setNear(str): A reference location area from where tweets were generated.(Name woks but longi latitude works better try it out)
setWithin (str): A distance radius from "near" location (e.g. 15mi).
setMaxTweets (int): The maximum number of tweets to be retrieved. If this number is unsetted or lower than 1 all possible tweets will be retrieved.(We need a lot of tweets . two datasets will be created test and train .)

We can retrieve all these data points from each tweet
id (str)  !!NEEDED!!
permalink (str)
username (str)
to (str)
text (str) !!NEEDED!!
date (datetime) in UTC !!NEEDED!!
retweets (int) !!NEEDED!!
favorites (int) 
mentions (str) 
hashtags (str) !!NEEDED!!
geo (str) !!NEEDED!!

once a database is ready , make sure theres a backup . Cuz we ll be running random things a lot . 
