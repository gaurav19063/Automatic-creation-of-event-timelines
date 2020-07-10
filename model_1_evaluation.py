#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
import contractions
from collections import Counter
from num2words import num2words
import numpy as np
import tweepy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[2]:


with open('tweet_timeline.pkl','rb') as f:
        tweet_timeline_dict = pickle.load(f)


# In[3]:


teams_hash_dict={"#mi":"mumbai indians","#csk":"chennai super kings","#rcb":"royal challengers bangalore","#kxip":"kings xi punjab",
            "#kkr":"kolkata knight riders","#dc":"delhi capitals","#srh":"sunrisers hyderabad","#rr":"rajasthan royals","#delhicapitals":"delhi capitals"}


# In[4]:


for key,val in tweet_timeline_dict.items():
    val=val.lower()
    for team in list(teams_hash_dict):
        val=re.sub("{} ".format(team),"{} ".format(teams_hash_dict[team]),val)
    tweet_timeline_dict[key]=val
    #print(key,val)


# In[5]:


#connection to Twitter API using tweepy

consumer_key = "CtmrTSUsnIRHKSgj4ktqK4NKb"
consumer_secret = "1MNfKmptTsrJcq7WMYxOoVwEgFCK5DNMQeC43IZDkPucPObCnZ"
access_key = "1231240089600057344-rqYnkeVvMaB1XwfvrERfI7aF38r69L"
access_secret = "KmRfAoexHxfkHKJaC3hwB0sgtfocyz58IqdzYiWeaXSGO"
# Authorization to consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Access to user's access key and access secret
auth.set_access_token(access_key, access_secret)

# Calling api
api = tweepy.API(auth)


# In[6]:


def get_username(username):
    # 200 tweets to be extracted
    #number_of_tweets=1
    try:
        tweets = api.user_timeline(screen_name=username)

        #tweets_for_csv = [tweet.text for tweet in tweets]  # CSV file created
        for tweet in tweets:
            #print(tweet)
            return tweet._json['user']['name']
    except tweepy.error.TweepError:
        return None


# In[7]:


teams_dict={"mi":"mumbai indians","csk":"chennai super kings","rcb":"royal challengers bangalore","kxip":"kings xi punjab",
            "kkr":"kolkata knight riders","dc":"delhi capitals","srh":"sunrisers hyderabad","rr":"rajasthan royals"}


# In[8]:


prefix_list=["mumbai","chennai","royal","kings","kolkata","delhi","sunrisers","rajasthan"]


# In[9]:


punct=string.punctuation
english_stopwords=stopwords.words('english')


# In[10]:


for key,val in tweet_timeline_dict.items():
        final_tokens = []

        #remove url from tweet text 
        tweet_text=re.sub(r"(http|https)\S+", "", val)
        tweet_text=re.sub(r'\B#\w*[a-zA-Z]+\w*',"",tweet_text)

        #Fix contractions
        tweet_text=contractions.fix(tweet_text)
        #print(tweet_text)


        #tokenize
        token_list=nltk.word_tokenize(tweet_text)
        i=0
        while i<len(token_list):
            if(token_list[i]=='@'): #replace twitter handle with screen name
                temp=token_list[i]+token_list[i+1]
                #print(temp)
                if get_username(temp) is not None:
                    #print(get_username(temp))
                    temp_list=nltk.word_tokenize(get_username(temp))
                    for t in temp_list:
                        final_tokens.append(t.lower())
                i=i+1
            #elif token_list[i].isnumeric(): #ignore digits from token list
            #    temp = num2words(token_list[i])
                #final_tokens.append(temp)
            #    i=i+1
            else:
                flag=1
                #remove punctuation from each string 
                str_temp = token_list[i].translate(str.maketrans('', '', punct))
                #print(str_temp)

                #convert to lower case
                str_temp=str_temp.lower()

                #replace team abbreviations with team name
                for team,team_name in teams_dict.items():
                    if str_temp==team:
                        temp_list = nltk.word_tokenize(team_name)
                        #print(temp_list)
                        for t in temp_list:
                             final_tokens.append(t)
                        flag=0
                if flag==0:
                    i=i+1
                    continue

                #stop word and punctuation tokens removal
                if str_temp not in english_stopwords:
                    #str_temp = str_temp.replace("'", "")
                    final_tokens.append(str_temp)
                i+=1
        tweet_timeline_dict[key]=final_tokens


# In[11]:


with open('model_1_evaluation_data.pkl','rb') as f:
        schedule_dict = pickle.load(f)


# In[12]:


matches=0
y_pred=[]
y_true=["Match" for x in range(len(list(schedule_dict)))]
for key,val in schedule_dict.items():
    intersect=set(val).intersection(set(tweet_timeline_dict[key]))
    if(len(intersect)<=3):
        if tweet_timeline_dict[key][0] in prefix_list or len(set(["win","won","romp","wins","match"]).intersection(intersect))>0:
            matches+=1
            y_pred.append("Match")
        else:
            y_pred.append("No Match")
    else:
        matches+=1
        y_pred.append("Match")
print("{}% match with ground truth".format(matches/len(list(schedule_dict))*100))


# In[13]:


print("Accuracy: ",accuracy_score(y_true, y_pred)*100,'%')


# In[14]:


print("Confusion Matrix: \n",confusion_matrix(y_true, y_pred))


# In[15]:


print("Classification Report: \n",classification_report(y_true, y_pred))

