#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from datetime import date, timedelta
import time
import math


# In[2]:


date_list=[]
start_date = date(2019, 3, 23)
end_date = date(2019, 5, 12)
delta = end_date - start_date
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    #print(day)
    #print(day+timedelta(days=1))
    date_list.append([day,day+timedelta(days=1)])


# In[3]:


with open('inverted_index_model_1.pkl','rb') as f:
        total_inverted_index = pickle.load(f)


# In[4]:


with open('tweet_dc_score_model_1.pkl','rb') as f:
        total_tweet_dc_score = pickle.load(f)


# In[5]:


with open('tweet_dict_model_1.pkl','rb') as f:
        tweet_dict = pickle.load(f)


# In[6]:


start = time.time()
print("TIMELINE")
print("---------------------------------------")
for d in date_list:
    tweet_score=[]
    tweet_score.append(0)
    inverted_index=total_inverted_index[d[0]]
    tweet_dc_score=total_tweet_dc_score[d[0]]
    for i in range(1,len(list(tweet_dc_score))):
        vector = []
        contextual_vector = []
        #construct document vector and contextual vector corresponding to vocabulary
        for word in list(set(inverted_index)):
            vector.append(inverted_index[word][i])
            contextual_vector.append(inverted_index[word][0])
        #compute cosine similarity between document vector and contextual vector
        #tweet_score.append(np.dot(vector, contextual_vector) / (np.sqrt(np.dot(vector,vector)) * np.sqrt(np.dot(contextual_vector,contextual_vector))))
        tweet_score.append(np.dot(vector, contextual_vector)/(np.linalg.norm(vector)*np.linalg.norm(contextual_vector)))

    #print(tweet_score[1:])

    i=1
    max_score=0
    max_index=0
    while i != len(tweet_score):
        #compute overall tweet score corresponding to each tweet
        #tweet_score[i]=0.5*tweet_score[i]+0.5*tweet_dc_score[i]
        tweet_score[i] = tweet_score[i]*math.log10(tweet_dc_score[i]+2)
        if(max_score<tweet_score[i]):
            max_score=tweet_score[i]
            max_index=i
        #print("{}:{}".format(tweet_dict[d[0]][i],tweet_score[i]))
        i=i+1

    #print("---------------------------------------")
    #print top tweet and date in the timeline
    print("{}".format(tweet_dict[d[0]][max_index]))
    print("---------------------------------------")
end = time.time()
print("Time taken = {} seconds".format(end-start))


# In[ ]:




