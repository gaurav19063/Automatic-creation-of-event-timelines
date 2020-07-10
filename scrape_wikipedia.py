import requests
import re
from bs4 import BeautifulSoup
import nltk
from datetime import datetime
import string
from nltk.corpus import stopwords
import pickle
from num2words import num2words


website_url=requests.get("https://www.hindustantimes.com/ipl/results/").text
soup = BeautifulSoup(website_url,"html.parser")
divs_result=soup.find_all('div',{"class":"match-result"})
divs_date=soup.find_all('div',{"class":"match-dt-tm"})

schedule_dict={}

for i in range(len(divs_result)):
    result=divs_result[len(divs_result)-i-1].text.strip()
    d=divs_date[len(divs_result)-i-1].text.lstrip()
    d=d.split(", ")[1].split()
    s = "{} {}, 2019".format(d[0], d[1])
    match_date = datetime.strptime(s, '%d %B, %Y').date()
    #match_date = d.strftime('%Y-%m-%d')
    schedule_dict.setdefault(match_date,'')
    schedule_dict[match_date]+=" "+result

s = "12 May, 2019".format(d[0], d[1])
schedule_dict[datetime.strptime(s, '%d %B, %Y').date()]="Mumbai Indians beat Chennai Super Kings by 1 run. Champions"

for key,val in schedule_dict.items():
    text = val.lower()
    tokenizer = nltk.tokenize.RegexpTokenizer((r'\d+\.\d+|\w+'))
    tokens = tokenizer.tokenize(text)
    if "abandoned" not in tokens:
        tokens=tokens+["win","won","romp","wins","match","table","top"]
    schedule_dict[key]=tokens
    print(key,schedule_dict[key])

print("----------------------------------------------")

website_url = requests.get("https://en.wikipedia.org/wiki/2019_Indian_Premier_League#Matches").text
soup = BeautifulSoup(website_url,"html.parser")
divs=soup.find_all('div',attrs={'style':'width: 100%; clear:both'})
english_stopwords=stopwords.words('english')
punct=string.punctuation

match_dict={}
replace_list=["(D/N)  Scorecard","(H)","16:00","20:00","20.00","19:30","M. Chinnaswamy Stadium","Feroz Shah Kotla",
              "Rajiv Gandhi International Cricket Stadium","Eden Gardens","Sawai Mansingh Stadium, Jaipur","Wankhede Stadium"
              "Punjab Cricket Association IS Bindra Stadium, Mohali","M. A. Chidambaram Stadium","Dr. Y. S. Rajasekhara Reddy ACA–VDCA Cricket Stadium, Visakhapatnam",
              "won the toss and elected to"]
for div in divs:
    text=div.text
    for item in replace_list:
        text=text.replace(item,"")
    text=re.sub(r"\[\d*\]","",text)
    text=text.lower()
    tokenizer = nltk.tokenize.RegexpTokenizer((r'\d+\.\d+|\w+'))
    tokens = tokenizer.tokenize(text)
    s = "{} {}, 2019".format(tokens[0],tokens[1])
    match_date = datetime.strptime(s, '%d %B, %Y').date()
    # match_date = d.strftime('%Y-%m-%d')
    tokens=tokens[2:]
    final_tokens=[]
    flag=0
    for token in tokens:
        if token == "umpires":
            flag=1
            continue
        if token == "player":
            flag=0
        if flag==1:
            continue
        if token.isalpha() and len(token)==1:
            continue
        elif token not in english_stopwords:
            for t in re.findall(r"[^\W\d_]+|\d+", token):
                final_tokens.append(t)
    if match_date in match_dict.keys():
        match_dict[match_date]+=list(set(final_tokens))
    else:
        match_dict[match_date]=list(set(final_tokens+schedule_dict[key]))

for key,val in match_dict.items():
    print("{}:{}".format(key,val))


f = open("model_1_evaluation_data.pkl", "wb")
pickle.dump(schedule_dict, f)
f.close()

f = open("model_2_evaluation_data.pkl", "wb")
pickle.dump(match_dict, f)
f.close()
