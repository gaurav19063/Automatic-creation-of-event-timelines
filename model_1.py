import math
import GetOldTweets3 as got
from datetime import date, timedelta
import tweepy
import nltk
import re
import string
from nltk.corpus import stopwords
import contractions
from collections import Counter
from num2words import num2words
import numpy as np
import time
import pickle

punct=string.punctuation
english_stopwords=stopwords.words('english')


teams_dict={"mi":"mumbai indians","csk":"chennai super kings","rcb":"royal challengers bangalore","kxip":"kings xi punjab",
            "kkr":"kolkata knight riders","dc":"delhi capitals","srh":"sunrisers hyderabad","rr":"rajasthan royals"}

'''
temp_dict={}


#generate team1vteam2 for matching hashtags such as #mivcsk
for k1 in list(teams_dict):
    for k2 in list(teams_dict):
        if k1!=k2:
            key="{}v{}".format(k1, k2)
            value="{} {}".format(teams_dict[k1],teams_dict[k2])
            temp_dict[key]=value

for key,val in temp_dict.items():
    teams_dict[key]=val
'''
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

'''
for key,val in teams_dict.items():
    print("{}:{}".format(key,val))
'''

print("TIMELINE")
print("--------------------------")


#returns Twitter name based on twitter handle
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



date_list=[]

#list of authoritative sources
source_list = ["@IPL", "@ESPNcricinfo", "@cricbuzz","@Cricketracker","@circleofcricket"]
#source_list=["@circleofcricket"]

#hashtag to be filtered
hashtags = ["IPL"]

#max number of tweets per source
num_tweets=20

#tweet range
start_date = date(2019, 3, 23)
end_date = date(2019, 5, 12)

#generate days between start and end date
delta = end_date - start_date
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    #print(day)
    #print(day+timedelta(days=1))
    date_list.append([day,day+timedelta(days=1)])

#returns contextual vector value for a token per day
def get_tf_idf_contextual(Alltoken_list_per_day, token_list_per_day,token):
    
    #compute df
    df=0
    for l in token_list_per_day:
        if token in l:
            df=df+1
    #list_freq = (Counter(Alltoken_list_per_day))
    # for token, value in list_freq.items():

    #compute tf
    tf=0
    for x in Alltoken_list_per_day:
        if x==token:
            tf=tf+1
    #print(tf,df,"here")
    
    #computing contextual vector score
    return 1+math.log10(tf)*math.log10(len(token_list_per_day)/df)
    #return 1+math.log10(tf)*math.log10(df)
    #return tf*df

tweet_dict={}
contextual_dict={}
tweet_timeline_dict={}

total_tweet_dc_score={}
total_inverted_index={}

for d in date_list:
    tweet_dc_score = {}
    inverted_index = {}
    temp_dict = {}
    all_token_list_per_day=[]
    token_list_per_day = []
    tweetID=1
    for source in source_list:
        temp_tweet_dc_score={}
        max_retweet_score=0
        max_likes_score=0
        for hashtag in hashtags:
            # print(d[0])
            #print(source)
            #print(hashtag)

            start = time.time()

            #set tweet criteria
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch("".format(hashtag))\
                                                       .setSince("{}".format(d[0]))\
                                                       .setUntil("{}".format(d[1]))\
                                                       .setLang("en")\
                                                       .setUsername("{}".format(source))\
                                                       .setTopTweets(num_tweets)\
                                                       .setMaxTweets(num_tweets)
            
            #fetch tweets based on criteria
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)

            #end = time.time()
            #print("{} sec".format(end - start))
            #start = time.time()

            #counter=0

            #for tweet in tweets:
            #    if hashtag in tweet.text:
            #        counter+=1

            #print(counter)

            for tweet in tweets:
                if hashtag in tweet.text:
                    final_tokens = []
                    
                    #remove url from tweet text 
                    tweet_text=re.sub(r"(http|https)\S+", "", tweet.text)
                    #tweet_text=re.sub(r'\B#\w*[a-zA-Z]+\w*',"",tweet_text)
                    
                    #Fix contractions
                    tweet_text=contractions.fix(tweet_text)
                    #print(tweet_text)
                    
                    if tweet.retweets > 0:
                        #print("---------------")
                        #print(tweet.username)
                        #print(tweet.date)
                        
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
                                if str_temp not in english_stopwords and len(str_temp)>1 and not str_temp.isnumeric():
                                    #str_temp = str_temp.replace("'", "")
                                    final_tokens.append(str_temp)
                            i=i+1
                        #print(tweet.text)
                        #print(tweet.favorites)
                        #print(tweet.retweets)
                        #print("---------------")

                        #maintain tweet with tweet ID in a dictionary
                        temp_dict[tweetID]=tweet_text

                        #tweet_score=math.log10(0.7*tweet.retweets+0.3*tweet.favorites+2)
                        #tweet_dc_score[tweetID]=tweet_score

                        #create temp dictionary storing tweet retweet and tweet score
                        temp_tweet_dc_score[tweetID]=[tweet.retweets,tweet.favorites]
                        max_retweet_score=max(max_retweet_score,tweet.retweets)
                        max_likes_score=max(max_likes_score,tweet.favorites)


                        #print(final_tokens)

                        #create inverted index for tokens - index 0 for df and word count of token corresponding to tweet ID in the remaining part 
                        for new_token in final_tokens:
                            if new_token in inverted_index.keys():
                                inverted_index[new_token][tweetID] = inverted_index[new_token][tweetID] + final_tokens.count(new_token)
                                inverted_index[new_token][0]+=1
                            else:
                                inverted_index[new_token] = [0 for x in range(len(source_list)*num_tweets)]
                                inverted_index[new_token][0]=1
                                inverted_index[new_token][tweetID]=final_tokens.count(new_token)

                        tweetID+=1
                    all_token_list_per_day = all_token_list_per_day + final_tokens
                    token_list_per_day.append(final_tokens)
            end = time.time()

        #update normalized tweet score per source
        for key,val in temp_tweet_dc_score.items():
            tweet_dc_score[key]=0.7*(val[0]/max_retweet_score)+0.3*(val[1]/max_likes_score)

    tweet_dict[d[0]]=temp_dict

    all_token_list_per_day_set=set(all_token_list_per_day) #vocabulary

    #return contextual vector word for each token
    contextual_dict_tokens = {}
    for token in set(all_token_list_per_day_set):
        contextual_dict_tokens[token] = get_tf_idf_contextual(all_token_list_per_day, token_list_per_day, token)

    #sort contextual tokens
    contextual_sorted_tokens = sorted(contextual_dict_tokens.items(), key=lambda x: x[1], reverse=True)
    
    #pick top 10 entries from contextual tokens
    contextual_list = []
    for i,t in enumerate(set(contextual_sorted_tokens)):
        if i==10:
            break
        contextual_list.append(t[0])

    contextual_dict_tokens[d[0]]=contextual_list

        #print("{},{}".format(t[0],t[1]))
    #print(contextual_sorted_tokens[:15])

    contextual_dict[d[0]]=contextual_sorted_tokens[:10]

    #print(d[0],contextual_sorted_tokens[:10])

    for key,val in inverted_index.items():
        idf = math.log10(tweetID/inverted_index[key][0])
        #df=inverted_index[key][0]
        for i in range(tweetID):
            if i!=0:
                inverted_index[key][i]=(math.log10(1+inverted_index[key][i]))*idf #calculate score for inverted index using tf*idf
                #inverted_index[key][i] = inverted_index[key][i] * df
                #inverted_index[key][i] = (math.log2(1 + inverted_index[key][i])) * df

        #update 0th index of inverted index with contextual score of the token    
        if key in set(contextual_list):
            inverted_index[key][0]=contextual_dict_tokens[key]


    #for key,val in inverted_index.items():
    #    print("{}:{}".format(key, val))

    tweet_score=[]
    tweet_score.append(0)
    for i in range(1,tweetID):
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
    tweet_timeline_dict[d[0]]=tweet_dict[d[0]][max_index]
    print("{}".format(tweet_dict[d[0]][max_index]))
    #print("---------------------------------------")

    total_tweet_dc_score[d[0]]=tweet_dc_score
    total_inverted_index[d[0]]=inverted_index


'''
f = open("tweet_timeline.pkl", "wb")
pickle.dump(tweet_timeline_dict, f)
f.close()

f = open("contextual_vector.pkl", "wb")
pickle.dump(contextual_dict, f)
f.close()
'''

f = open("tweet_dict_model_1.pkl", "wb")
pickle.dump(tweet_dict, f)
f.close()

f = open("tweet_dc_score_model_1.pkl", "wb")
pickle.dump(total_tweet_dc_score, f)
f.close()

f = open("inverted_index_model_1.pkl", "wb")
pickle.dump(total_inverted_index, f)
f.close()


