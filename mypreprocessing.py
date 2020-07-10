import re
import string
from nltk.corpus import stopwords
import contractions
import nltk
from num2words import num2words
import tweepy

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

english_stopwords=stopwords.words('english')

def preprocess(temp):
    # expand using contractions
    temp=re.sub(r"(http|https)\S+", "", temp)
    temp = contractions.fix(temp)
    # tokenize
    tokens=nltk.word_tokenize(temp)
    #tokens = tokenizer.tokenize(temp)

    # for i,token in enumerate(tokens):
    #    if token[0].isupper():
    #        print("{}:{}:{}".format(i,file,token))

    string.punctuation = string.punctuation + "''``--"

    #print(tokens)

    new_tokens = []
    pattern = ('\d+(\.\d+)?')

    for i, token in enumerate(tokens):
        if i==len(tokens):
            break
        if(token=='@'): 
            #replace twitter handle with screen name
            temp=tokens[i]+tokens[i+1]
            #print(temp)
            if get_username(temp) is not None:
                #print(get_username(temp))
                temp_list=nltk.word_tokenize(get_username(temp))
                for t in temp_list:
                    new_tokens.append(t.lower())
            i=i+2
            continue
        if len(token)<3:
            continue
        if token in english_stopwords:
           continue
        if (token not in string.punctuation):
            if not re.match(pattern, token):
                for s in token:
                    if s in string.punctuation:
                        token = token.replace(s, '')
            else:
                if token.isdigit():
                    token = num2words(float(token))
            new_tokens.append(token.lower().encode("ascii", errors="ignore").decode())

    '''
        # lemmitization
        lemmatizer = nltk.WordNetLemmatizer()
        final_tokens=[]
        for token in new_tokens:
            final_tokens.append(lemmatizer.lemmatize(token))
  
    # stemmer
    stemmer = nltk.PorterStemmer()
    final_tokens = []
    for token in new_tokens:
        final_tokens.append(stemmer.stem(token))
    '''

    return new_tokens
