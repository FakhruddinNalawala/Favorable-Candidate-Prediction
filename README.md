# Election
Okay so this will be the working directory for our project. 

TWEETS.PY  </br>
This is the script that will scrape tweets from twitter .   
Set up a mongodb database and query the results into the database .   
We cant set all these fields into the querysearch using this Library . Use as Required .    

setUsername (str or iterable): An optional specific username(s) from a twitter account (with or without "@").   </br>
setSince (str. "yyyy-mm-dd"): A lower bound date (UTC) to restrict search.  </br>
setUntil (str. "yyyy-mm-dd"): An upper bound date (not included) to restrict search.</br>  
setQuerySearch (str): A query text to be matched.</br>
setTopTweets (bool): If True only the Top Tweets will be retrieved.</br>
setNear(str): A reference location area from where tweets were generated.(Name woks but longi latitude works better try it out)</br>
setWithin (str): A distance radius from "near" location (e.g. 15mi).</br>
setMaxTweets (int): The maximum number of tweets to be retrieved. If this number is unsetted or lower than 1 all possible tweets will be retrieved.(We need a lot of tweets . two datasets will be created test and train .)</br>

We can retrieve all these data points from each tweet</br>
id (str)  !!NEEDED!!</br>
permalink (str)</br>
username (str)</br>
to (str)</br>
text (str) !!NEEDED!!</br>
date (datetime) in UTC !!NEEDED!!</br>
retweets (int) !!NEEDED!!</br>
favorites (int) </br>
mentions (str) </br>
hashtags (str) !!NEEDED!! </br>
geo (str) !!NEEDED!!</br>

once a database is ready , make sure theres a backup . Cuz we ll be running random things a lot . 
