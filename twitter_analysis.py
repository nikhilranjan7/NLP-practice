#!/usr/bin/python
# -*- coding: utf-8 -*-
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

# consumer key, consumer secret, access token, access secret.

ckey = '5WTo3uWBaiisgkpfj4Uxo93z3'
csecret = 'x1DJU6U2ZaWolxjDax0wpoI6vMmJsTwSI0070NWzuYj4dKHUGY'
atoken = '70351928-5fdWGmR6sw4oorlZtB0N0jHfh56dYzEBejLvLKFaz'
asecret = '4nGh4GKrRe839V8j91qrtrZMLxSGNL1UGqorx7K671iP8'


class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        tweet = all_data['text']
        (sentiment_value, confidence) = s.sentiment(tweet)
        print (tweet, sentiment_value, confidence)

        if confidence * 100 >= 80:
            output = open('twitter-out.txt', 'a')
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print (status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=['rock'])
