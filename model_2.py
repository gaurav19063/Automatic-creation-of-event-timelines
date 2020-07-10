#!/usr/bin/env python
# coding: utf-8

# In[1]:


import GetOldTweets3 as got
from datetime import date, timedelta
import tweepy
import nltk
import re
import string
from nltk.corpus import stopwords
import contractions
import time
import math
from collections import Counter
import pickle
import pandas as pd
import numpy as np
import math


# In[2]:


punct = string.punctuation
english_stopwords = stopwords.words('english')

teams_dict={"mi":"mumbai indians","csk":"chennai super kings","rcb":"royal challengers bangalore","kxip":"kings xi punjab",
            "kkr":"kolkata knight riders","dc":"delhi capitals","srh":"sunrisers hyderabad","rr":"rajasthan royals"}

# connection to Twitter API using tweepy

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


# In[3]:


# returns Twitter name based on twitter handle
def get_username(username):
    # 200 tweets to be extracted
    # number_of_tweets=1
    try:
        tweets = api.user_timeline(screen_name=username)

        # tweets_for_csv = [tweet.text for tweet in tweets]  # CSV file created
        for tweet in tweets:
            # print(tweet)
            return tweet._json['user']['name']
    except tweepy.error.TweepError:
        return None


# In[4]:


date_list = []

# list of authoritative sources
source_list = ["@IPL", "@ESPNcricinfo", "@cricbuzz", "@Cricketracker", "@circleofcricket"]
#source_list = ["@ESPNcricinfo", "@cricbuzz", "@Cricketracker", "@circleofcricket"]
# source_list=["@circleofcricket"]

# hashtag to be filtered
hashtags = ["IPL"]

# max number of tweets per source
num_tweets = 20

# tweet range
start_date = date(2019, 3, 23)
end_date = date(2019, 4, 16)

# generate days between start and end date
delta = end_date - start_date
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    # print(day)
    # print(day+timedelta(days=1))
    date_list.append([day, day + timedelta(days=1)])
print(date_list)


# In[5]:


team_hashtag_dict={}
#generate team1vteam2 for matching hashtags such as #mivcsk
for k1 in list(teams_dict):
    for k2 in list(teams_dict):
        if k1!=k2:
            key="{}v{}".format(k1, k2)
            value="{} {}".format(teams_dict[k1],teams_dict[k2])
            team_hashtag_dict[key]=value
#print(team_hashtag_dict)


# In[6]:


tweet_dict = {}
tweet_tokens_dict={}
tokens_df_dict = {}
final_token_list=[]

tweetID = 0

for d in date_list:
    for source in source_list:
        max_tweet_score=0
        max_like_score=0
        startID=tweetID
        for hashtag in hashtags:
            print(d[0])
            print(source)
            print(hashtag)

            start = time.time()

            # set tweet criteria
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch("".format(hashtag))\
                                                       .setSince("{}".format(d[0]))\
                                                       .setUntil("{}".format(d[1]))\
                                                       .setLang("en")\
                                                       .setUsername("{}".format(source))\
                                                       .setTopTweets(num_tweets)\
                                                       .setMaxTweets(num_tweets)

            # fetch tweets based on criteria
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)

            # end = time.time()
            # print("{} sec".format(end - start))
            # start = time.time()

            # counter=0

            # for tweet in tweets:
            #    if hashtag in tweet.text:
            #        counter+=1

            # print(counter)

            for tweet in tweets:
                if hashtag in tweet.text:
                    final_tokens = []

                    # remove url from tweet text
                    tweet_text = re.sub(r"(http|https)\S+", "", tweet.text)
                    #tweet_text=re.sub(r'\B#\w*[a-zA-Z]+\w*',"",tweet_text)

                    # Fix contractions
                    tweet_text = contractions.fix(tweet_text)
                    #print(tweet_text)

                    if tweet.retweets > 0:
                        # print("---------------")
                        # print(tweet.username)
                        # print(tweet.date)

                        # tokenize
                        token_list = nltk.word_tokenize(tweet_text)
                        i = 0
                        while i < len(token_list):
                            if (token_list[i] == '@'):  # replace twitter handle with screen name
                                temp = token_list[i] + token_list[i + 1]
                                # print(temp)
                                if get_username(temp) is not None:
                                    # print(get_username(temp))
                                    temp_list = nltk.word_tokenize(get_username(temp))
                                    for t in temp_list:
                                        final_tokens.append(t.lower())
                                i = i + 2
                            # elif token_list[i].isnumeric(): #ignore digits from token list
                            #    temp = num2words(token_list[i])
                            # final_tokens.append(temp)
                            #    i=i+1
                            else:
                                flag = 1
                                # remove punctuation from each string
                                str_temp=token_list[i].translate(str.maketrans('', '', punct))
                                # print(str_temp)

                                # convert to lower case
                                str_temp = str_temp.lower()

                                # replace team abbreviations with team name
                                for team, team_name in teams_dict.items():
                                    if str_temp == team:
                                        temp_list = nltk.word_tokenize(team_name)
                                        # print(temp_list)
                                        for t in temp_list:
                                            final_tokens.append(t)
                                        flag = 0
                                if flag == 0:
                                    i = i + 1
                                    continue
                                    
                                for team, team_name in team_hashtag_dict.items():
                                    if str_temp == team:
                                        temp_list = nltk.word_tokenize(team_name)
                                        # print(temp_list)
                                        for t in temp_list:
                                            final_tokens.append(t)
                                        flag = 0
                                if flag == 0:
                                    i = i + 1
                                    continue
                                    
                                if str_temp.isnumeric() or str_temp in punct:
                                    #print(str_temp)
                                    i=i+1
                                    continue

                                # stop word and punctuation tokens removal
                                if str_temp not in english_stopwords:
                                    # str_temp = str_temp.replace("'", "")
                                    final_tokens.append(str_temp)
                            i = i + 1
                        # print(tweet.text)
                        # print(tweet.favorites)
                        # print(tweet.retweets)
                        # print("---------------")
                        
                        max_tweet_score=max(max_tweet_score,tweet.retweets)
                        max_like_score=max(max_like_score,tweet.favorites)

                        # maintain tweet with tweet ID in a dictionary
                        tweet_dict[tweetID] = [tweet_text,0,tweet.retweets,tweet.favorites]

                        tweet_tokens_dict[tweetID]=final_tokens

                        #print(final_tokens)
                        final_token_list=list(set(final_token_list+final_tokens))

                        for token in set(final_tokens):
                            tokens_df_dict.setdefault(token,0)
                            tokens_df_dict[token]+=1
                            
                        tweetID += 1
            end = time.time()
        while startID!=tweetID:
            #print(startID,tweet_dict[startID])
            tweet_dict[startID][1]=((0.7*tweet_dict[startID][2]/max_tweet_score)+(0.3*tweet_dict[startID][3]/max_like_score))
            #print(max_tweet_score,max_like_score)
            #print(tweet_dict[startID][1])
            startID+=1
        print("-----------------------------------------")


# In[7]:


vocab_size=len(set(final_token_list))
vocab=sorted(set(final_token_list))


# In[8]:


tweet_df=pd.DataFrame(columns=vocab)

for key, val in tweet_tokens_dict.items():
    row=[0 for x in range(vocab_size)]
    final_tokens=Counter(tweet_tokens_dict[key])
    for i,term in enumerate(vocab):
        if term in set(tweet_tokens_dict[key]):
            idf = math.log10(tweetID / tokens_df_dict[term])
            row[i]=final_tokens[term]*idf
    tweet_df = tweet_df.append(pd.Series(row, index=tweet_df.columns), ignore_index=True)


# In[9]:


print(tweet_df.shape)


# In[10]:


def cosine_similarity(vect1,vect2):
    return np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))


# In[11]:


def jaccard_similarity(list1,list2):
    doc_intersection=len(set(list1).intersection(set(list2)))
    return (doc_intersection)/(len(set(list1))+len(set(list2))-doc_intersection)


# In[12]:


with open('contextual_vector.pkl','rb') as f:
        contextual_vector_dict = pickle.load(f)


# In[13]:


common_terms=['win','won','romp','wins','team','season','game','like','toss','wicket','match','xi','vs','runs','four','six','dot','ball','overs','innings','vivoipl','ipl','balls','run']


# In[14]:


with open('model_1_evaluation_data.pkl','rb') as f:
        schedule_dict = pickle.load(f)


# In[15]:


for key,val in contextual_vector_dict.items():
    temp=[]
    for v in val:
        if v[0]=="raina🇮🇳":
            temp.append("raina")
        elif v[0] not in common_terms:
            temp.append(v[0])
    if key in list(schedule_dict):
        #print(temp)
        val=schedule_dict[key]
        for v in val:
            if len(v)>1 and v not in common_terms and v not in temp:
                    temp.append(v)
    contextual_vector_dict[key]=temp


# In[16]:


for key,val in contextual_vector_dict.items():
    print(key,val)


# In[17]:


tweet_list=[key for key,val in sorted(dict(filter(lambda x:x[1][1]>=0.1,tweet_dict.items())).items(),key=lambda x: x[1][1],reverse=True)]
tweet_list_copy=[key for key,val in sorted(dict(filter(lambda x:x[1][1]>=0.1,tweet_dict.items())).items(),key=lambda x: x[1][1],reverse=True)]
clus_dict={}
#print(tweet_list)
for tweetID1 in tweet_list:
    if len(tweet_list_copy)==0:
        break
    if tweetID1 in set(tweet_list_copy):
        clus_dict.setdefault(tweetID1,[])
        #print(tweetID1,tweet_dict[tweetID1][0],tweet_dict[tweetID1][1])
        for tweetID2 in tweet_list:
            if tweetID2 in set(tweet_list_copy):
                if tweetID1!=tweetID2:
                    cosine_score=cosine_similarity(tweet_df.loc[tweetID1],tweet_df.loc[tweetID2])
                    #jaccard_score=jaccard_similarity(tweet_tokens_dict[tweetID1],tweet_tokens_dict[tweetID2])
                    if cosine_score>0.1:
                        if(len(clus_dict[tweetID1])<=len(source_list)):
                            clus_dict[tweetID1].append(tweetID2)
                            tweet_list_copy.remove(tweetID2)
        tweet_list_copy.remove(tweetID1)
        #print(clus_dict[tweetID1])
        #print(tweet_list)
        #print("---------------------------------------------")


# In[18]:


#tweet_list=[key for key,val in sorted(dict(filter(lambda x:x[1][1]>=0.5,tweet_dict.items())).items(),key=lambda x: x[1][1],reverse=True)]
index=0
clus_dict_copy={}
tweet_timeline_2_dict={}
for key,val in clus_dict.items():
    clus_dict_copy[key]=val
temp=[]
for key,val in contextual_vector_dict.items():
    #print(val)
    index=index+1
    #if index!=len(list(contextual_vector_dict)):
    if index>25:
          break
    contextual_vector=[1 for x in range(len(val))]
    max_score=0
    tweet_list=list(clus_dict_copy)
    best_tweet_ID=tweet_list[0]
    for tweetID in set(tweet_list):
        vector=[0 for x in range(len(val))]
        tokens_list=tweet_tokens_dict[tweetID]
        if "raina🇮🇳" in tokens_list:
            tokens_list[tokens_list.index("raina🇮🇳")]="raina"
        for i,v in enumerate(val):
            if v in set(tokens_list):
                vector[i]=1
        cosine_score=0
        total_score=0
        if sum(vector)>2:
            cosine_score=cosine_similarity(contextual_vector,vector) 
            total_score=cosine_score#+0.5*tweet_dict[tweetID][1]
            #if cosine_score>=0.3:
        
        '''
        print(val)
        print(tweet_tokens_dict[tweetID])
        print(vector)
        print("Cosine score:",cosine_score)
        print("Tweet quality score:",tweet_dict[tweetID][1])
        print(tweet_dict[tweetID][0])
        print(total_score)
        print("---------------------------")
        '''

        if max_score<total_score:
            max_score=total_score
            best_tweet_ID=tweetID
            temp=vector
    del clus_dict_copy[best_tweet_ID]
    print(key,tweet_dict[best_tweet_ID][0],val,temp,max_score,tweet_dict[best_tweet_ID][1])
    tweet_timeline_2_dict[key]=tweet_dict[best_tweet_ID][0]
    print("---------------------------")
    #print(val)
    #print(tweet_tokens_dict[best_tweet_ID])


# In[712]:


f = open("tweet_timeline_2.pkl", "wb")
pickle.dump(tweet_timeline_2_dict, f)
f.close()


# In[714]:


tweet_df.to_csv("model_2_vector.csv")


# In[715]:


f = open("tweet_dict_model_2.pkl", "wb")
pickle.dump(tweet_dict, f)
f.close()


# In[716]:


f = open("tweet_tokens_dict_model_2.pkl", "wb")
pickle.dump(tweet_tokens_dict, f)
f.close()


# In[19]:


f = open("cluster_dict_model_2.pkl", "wb")
pickle.dump(clus_dict, f)
f.close()


# In[717]:


i=1
for key,val in clus_dict.items():
    if len(val)>=len(source_list):
        print("Cluster {}".format(i))
        print(tweet_dict[key][0],tweet_dict[key][1])
        for v in val:
            print(tweet_dict[v][0],tweet_dict[v][1])
        i=i+1
        print()