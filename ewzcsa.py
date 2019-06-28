# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:28:41 2019

@author: Terence.Tachiona
"""

import twee

py           
import pandas as pd     
import numpy as np     
import plotly
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from credentials import *
import tweepy 
from textblob import TextBlob     
import re                        
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True) 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
print('Imported all libraries')
def twitter_setup():

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    
    api = tweepy.API(auth)
    return api

extractor = twitter_setup()

def get_user_tweets(api, username):
        """Return a list of all tweets from the authenticated API"""
        tweets = []
        for status in tweepy.Cursor(api.user_timeline, screen_name=username).items():
            tweets.append(status)
        return tweets 

alltweets = get_user_tweets(extractor, 'econet_support')

print("Number of tweets extracted: {}.\n".format(len(alltweets)))

print("600 recent tweets:\n")
for tweet in alltweets[:600]:
    print(tweet.text)
    print()

data = pd.DataFrame(data=[tweet.text for tweet in alltweets], columns=['Tweets'])

display(data.head(1000))

data.to_csv(r'C:\Users\terence.tachiona\Desktop\145191524ewzcsstweets.csv')

print(dir(alltweets[0]))
print(alltweets[0].id)
print(alltweets[0].created_at)
print(alltweets[0].source)
print(alltweets[0].favorite_count)
print(alltweets[0].retweet_count)
print(alltweets[0].geo)
print(alltweets[0].coordinates)
print(alltweets[0].entities)

data['len']  = np.array([len(tweet.text) for tweet in alltweets])
data['ID']   = np.array([tweet.id for tweet in alltweets])
data['Date'] = np.array([tweet.created_at for tweet in alltweets])
data['Source'] = np.array([tweet.source for tweet in alltweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in alltweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in alltweets])

display(data.head(10))

mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]


print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))


print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

tlen.plot(figsize=(16,4), color='r');

tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a Tweet by removing 
    links and special characters using regex re.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility TextBlob to classify the polarity of a Tweet
    using TextBlob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

display(data.head(100))

data.to_csv(r'C:\Users\terence.tachiona\Desktop\16519ewzcsstweets.csv')

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]


print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))

data['polarity'] = data['Tweets'].map(lambda text: TextBlob(text).sentiment.polarity)
data['review_len'] = data['Tweets'].astype(str).apply(len)
data['word_count'] = data['Tweets'].apply(lambda x: len(str(x).split()))

print('10 random responses with the highest positive sentiment polarity: \n')

cl = data.loc[data.polarity == 1, ['Tweets']].sample(10).values
for c in cl:
    print(c[0])
    
print('10 random responses with the most neutral sentiment(zero) polarity: \n')
cl = data.loc[data.polarity == 0, ['Tweets']].sample(10).values
for c in cl:
    print(c[0])
    

# =============================================================================
# print('10 responses with the most negative polarity: \n')
# cl = data.loc[data.polarity == -1, ['Tweets']].sample(10).values
# for c in cl:
#     print(c[0])
# =============================================================================
    

data['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')


#Distribution of Sentiment Polarity Scores
data['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

#Distribution of Responses Tweets rating
data['SA'].iplot(
    kind='hist',
    xTitle='rating',
    linecolor='black',
    yTitle='count',
    title='Response Tweet Rating Distribution')

#Distribution of Word Count
data['word_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Tweets Text Word Count Distribution')



data['SA'].value_counts().plot.bar(color = 'blue', figsize = (6, 4))

len_data = data['Tweets'].str.len().plot.hist(color = 'purple', figsize = (6, 4))

    # Time Series
time_likes = pd.Series(data=data['len'].values, index=data['Date'])
time_likes.plot(figsize=(16, 4), color='r')
plt.show()
    
time_favs = pd.Series(data=data['Likes'].values, index=data['Date'])
time_favs.plot(figsize=(16, 4), color='r')
plt.show()

time_retweets = pd.Series(data=data['RTs'].values, index=data['Date'])
time_retweets.plot(figsize=(16, 4), color='r')
plt.show()

    # Layered Time Series:
time_likes = pd.Series(data=data['Likes'].values, index=data['Date'])
time_likes.plot(figsize=(16, 4), label="Likes", legend=True)
time_retweets = pd.Series(data=data['RTs'].values, index=data['Date'])
time_retweets.plot(figsize=(16, 4), label="RTs", legend=True)
plt.show()