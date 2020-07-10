#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import time
import numpy as np


# In[2]:


source_list = ["@IPL", "@ESPNcricinfo", "@cricbuzz", "@Cricketracker", "@circleofcricket"]


# In[3]:


def cosine_similarity(vect1,vect2):
    return np.dot(vect1,vect2)/(np.linalg.norm(vect1)*np.linalg.norm(vect2))


# In[4]:


def jaccard_similarity(list1,list2):
    doc_intersection=len(set(list1).intersection(set(list2)))
    return (doc_intersection)/(len(set(list1))+len(set(list2))-doc_intersection)


# In[5]:


with open('contextual_vector.pkl','rb') as f:
        contextual_vector_dict = pickle.load(f)


# In[6]:


common_terms=['win','won','romp','wins','team','season','game','like','toss','wicket','match','xi','vs','runs','four','six','dot','ball','overs','innings','vivoipl','ipl','balls','run']


# In[7]:


with open('model_1_evaluation_data.pkl','rb') as f:
        schedule_dict = pickle.load(f)


# In[8]:


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


# In[9]:


tweet_df = pd.read_csv("model_2_vector.csv") 


# In[10]:


with open('tweet_dict_model_2.pkl','rb') as f:
        tweet_dict = pickle.load(f)


# In[11]:


with open('tweet_tokens_dict_model_2.pkl','rb') as f:
        tweet_tokens_dict = pickle.load(f)


# In[12]:


with open('cluster_dict_model_2.pkl','rb') as f:
        clus_dict = pickle.load(f)


# In[13]:


#tweet_list=[key for key,val in sorted(dict(filter(lambda x:x[1][1]>=0.5,tweet_dict.items())).items(),key=lambda x: x[1][1],reverse=True)]
index=0
clus_dict_copy={}
start = time.time()
print("TIMELINE")
print("---------------------------")
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
    print(tweet_dict[best_tweet_ID][0])
    print("----------------------------------")
    #print(val)
    #print(tweet_tokens_dict[best_tweet_ID])
end = time.time()
print("Time taken = {} seconds".format(end-start))

