#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
hotel = pd.read_csv("Hard_Rock_New_Review.csv", engine='python')


# In[2]:


import nltk
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk import FreqDist
nltk.download('stopwords') # run this one time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib as mpl
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
nltk.download('wordnet')
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet


# # exploratory data analysis

# In[3]:


print(hotel.describe().transpose())


# In[4]:


hotel.head()


# In[5]:


hotel[hotel['Review'].isnull()]


# In[6]:


hotel['Rating'] = hotel['Rating'].replace({10:'VERY NEGATIVE',20:'NEGATIVE',30:'NEUTRAL',40:'POSITIVE',50:'VERY POSITIVE'})


# # function to plot most frequent terms

# In[7]:



def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


# In[8]:


freq_words(hotel['Review'])


# # lenghth of review

# In[9]:


hotel_1 = hotel.copy()
hotel_1["Review"] = hotel_1["Review"].apply(str)


# In[10]:


hotel_1["REVIEWS LENGTH"] = hotel_1["Review"].apply(lambda w : len(re.findall(r'\w+', w)))


# In[11]:


hotel_1.sample(20)


# In[12]:


hotel_1["REVIEWS LENGTH"].describe()


# In[13]:


hotel_1["REVIEWS LENGTH"].sum() # TOTAL NUMBERS OF WORDS IN REVIEWS IS 95653


# In[14]:


hotel_1["Review"].describe()


# In[15]:


#plt.figure()
sns.boxplot(data = hotel_1, x = "REVIEWS LENGTH")
plt.xlabel('Number of Words')
plt.title('Review Length, Including Stop Words')
plt.show()


# In[16]:


sns.distplot(hotel_1["REVIEWS LENGTH"], kde = False)
plt.xlabel('Distribution of Review Length')
plt.title('Review Length, Including Stop Words')
plt.show()


# In[17]:


x_rating = hotel.Rating.value_counts()
y_rating = x_rating.sort_index()
plt.figure(figsize=(50,30))
sns.barplot(x_rating.index, x_rating.values, alpha=0.8)
plt.title("Rating Distribution", fontsize=50)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.xlabel('Employee Ratings', fontsize=50)
plt.xticks(fontsize=40)
plt.show()


# In[18]:


plt.figure(figsize=(30,10))
plt.title('Percentage of Ratings', fontsize=20)
hotel.Rating.value_counts().plot(kind='pie', labels=['VERY POSITIVE','POSITIVE','NETURAL','NEGATIVE','VERY NEGATIVE'],
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
plt.ylabel("Rating",fontsize=40)
plt.show()


# In[19]:


date=hotel['Date']
hotel.date = date.astype("datetime64")
hotel.groupby(hotel.date.dt.month).count().plot(kind="bar")


# In[20]:


hotel.drop(['Unnamed: 0','Date','Rating','Title'],axis=1,inplace=True)


# In[21]:


hotel.head()


# # cleaning the text
# 

# In[22]:


import nltk
nltk.download('stopwords')


# In[23]:


hotel_df = hotel.copy()
stop_words = stopwords.words("english")


# In[24]:


def clean(s):
    s = s.lower()                   #Converting to lower case
    s = re.sub(r'[^\w\s]', ' ', s)  #Removing punctuation
    s = re.sub(r'[\d+]', ' ', s)    #Removing Numbers
    s = s.strip()                   #Removing trailing spaces
    s = re.sub(' +', ' ', s)        #Removing extra whitespaces
    return s


# In[25]:


hotel_df["Review"] = hotel_df["Review"].apply(lambda x: clean(x))


# In[26]:


hotel_df.head(20)


# # stopwards

# In[27]:


hotel_df["Review"] = hotel_df["Review"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))


# In[28]:


hotel_df["Review"][2]


# # stemming

# In[29]:



st = PorterStemmer()
hotel_df['Review'] = hotel_df['Review'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
hotel_df['Review']


# # world cloud

# In[30]:


reviews = pd.Series(hotel_df["Review"].tolist()).astype(str)
plt.figure(figsize = (9, 9))
rev_wcloud_all = WordCloud(width = 900, height = 900, colormap = 'plasma', max_words = 150).generate(''.join(reviews))
plt.imshow(rev_wcloud_all)
plt.tight_layout(pad = 0.2)
plt.axis('off')
plt.show()


# In[31]:


#pip install textblob      

import textblob            
from textblob import TextBlob


# # TOKENIZATION
# 
# #### Tokenization is splitting a body of text into smaller units, such as individual words or terms. Each of these smaller units are called tokens

# In[32]:


from nltk.tokenize import word_tokenize, RegexpTokenizer


# In[33]:


hotel_df['Review']=hotel_df['Review'].apply(str)


# In[34]:


tokenizer = RegexpTokenizer(r'\w+')
hotel_df["Reviews_Token"] = hotel_df["Review"].apply(lambda x: tokenizer.tokenize(x))


# # LEMMATIZATION
# 
# ##### Lemmatization extracts the root of the word. For example, from the word "driving', "drive" is extracted. Unlike stemming, lemmatization understands the context and provides the root words rather than simply removing the suffix or prefix of the word.

# In[35]:


lemm = WordNetLemmatizer()


# In[36]:


def to_wordnet(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemm_text(text):
    nltk_tagged = nltk.pos_tag(text)
    # Result = (text, pos_tag)
    wordnet_tagged = map(lambda x: (x[0], to_wordnet(x[1])), nltk_tagged)
    lemm_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemm_sentence.append(word)
        else:
            lemm_sentence.append(lemm.lemmatize(word, tag))
    return lemm_sentence


# In[37]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[38]:


hotel_df["Reviews_Lemm"] = hotel_df["Reviews_Token"].apply(lambda x: lemm_text(x))


# In[39]:


hotel_df[["Reviews_Token", "Reviews_Lemm"]].sample(14)


# # most common words 

# In[41]:


import itertools
import collections


# In[42]:


review_list = list(itertools.chain.from_iterable(hotel_df['Reviews_Lemm']))
rev_word_freq = collections.Counter(review_list)

word_freq_DF = pd.DataFrame(rev_word_freq.most_common(15), columns=['Words', 'Count'])
word_freq_DF


# In[43]:


sns.barplot(data = word_freq_DF, x = "Words", y = "Count")
plt.ylabel("Frequency")
plt.xticks(rotation = 90)
plt.title("15 Most Frequent Words in Hard Rock Cafe NY Reviews")
plt.show()


# # SENTIMENT ANALYSIS

# In[44]:


hotel_df["Sentiment_TextBlob"] = hotel_df["Review"].apply(lambda x: TextBlob(x).sentiment[0])


# In[45]:


hotel_df[["Review", "Sentiment_TextBlob"]].head(10)


# In[49]:


from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[50]:


senti = SentimentIntensityAnalyzer()
hotel_df["Sentiment_VADER"] = hotel_df["Review"].apply(lambda x: senti.polarity_scores(x)['compound'])


# In[51]:


hotel_df[["Review", "Sentiment_VADER", "Sentiment_TextBlob"]].sample(15)


# In[52]:


hotel_df[["Review", "Sentiment_VADER"]][hotel_df["Sentiment_VADER"] < -0.1]


# In[53]:


sns.distplot(hotel_df["Sentiment_VADER"], kde = False, label = "VADER")
sns.distplot(hotel_df["Sentiment_TextBlob"], kde = False, label = "TextBlob")
plt.xlabel("Sentiment Polarity Scores")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[54]:


sns.boxplot(data = pd.melt(hotel_df[["Sentiment_VADER", "Sentiment_TextBlob"]]), x = "value", y = "variable")
plt.xlabel("Sentiment Polarity Scores")
plt.ylabel("Type of Sentiment Analysis")
plt.show()


# # MOST POSITIVE REVIEWS - TEXTBLOB

# In[55]:


DF_Pos_TB = hotel_df[["Review","Reviews_Lemm","Sentiment_TextBlob"]][hotel_df["Sentiment_TextBlob"] == hotel_df["Sentiment_TextBlob"].max()]


# In[56]:


DF_Pos_TB


# # WORDCLOUD OF THE MOST POSITIVE REVIEWS - TEXTBLOB

# In[57]:


# Word Cloud from the most postive reviews from TextBlob

plt.figure(figsize = (9, 9))
wcloud_Pos_TB = WordCloud(width = 900, height = 900, colormap = 'plasma', max_words = 150).generate(' '.join(pd.Series(DF_Pos_TB["Review"].tolist()).astype(str)))
plt.imshow(wcloud_Pos_TB)
plt.tight_layout(pad = 0.2)
plt.axis('off')
plt.show()


# # MOST POSTIVE REVIEWS - VADER

# In[58]:


hotel_df[["Review","Reviews_Lemm","Sentiment_VADER"]][hotel_df["Sentiment_VADER"] == hotel_df["Sentiment_VADER"].max()]


# In[59]:


DF_Pos_VADER = hotel_df[["Review","Reviews_Lemm","Sentiment_VADER"]][hotel_df["Sentiment_VADER"] > 0.8]


# In[60]:


DF_Pos_VADER


# # WORDCLOUD OF THE MOST POSTIVE REVIEWS - VADER

# In[61]:


plt.figure(figsize = (9, 9))
wcloud_Pos_VADER = WordCloud(width = 900, height = 900, colormap = 'plasma', max_words = 150).generate(' '.join(pd.Series(DF_Pos_VADER["Review"].tolist()).astype(str)))
plt.imshow(wcloud_Pos_VADER)
plt.tight_layout(pad = 0.2)
plt.axis('off')
plt.show()


# # MOST NEGATIVE REVIEWS - TEXTBLOB

# In[62]:


hotel_df[["Review","Reviews_Lemm","Sentiment_TextBlob"]][hotel_df["Sentiment_TextBlob"] == hotel_df["Sentiment_TextBlob"].min()]


# In[63]:


DF_Neg_TB = hotel_df[["Review","Reviews_Lemm","Sentiment_TextBlob"]][hotel_df["Sentiment_TextBlob"] < -0.0]


# In[64]:


DF_Neg_TB


# # WORDCLOUD OF THE MOST NEGATIVE REVIEWS - TEXTBLOB

# In[65]:


# Word Cloud of the most negative reviews from TextBlob

plt.figure(figsize = (9, 9))
wcloud_Neg_TB = WordCloud(width = 900, height = 900, colormap = 'plasma', max_words = 150).generate(' '.join(pd.Series(DF_Neg_TB["Review"].tolist()).astype(str)))
plt.imshow(wcloud_Neg_TB)
plt.tight_layout(pad = 0.2)
plt.axis('off')
plt.show()


# # MOST NEGATIVE REVIEWS - VADER

# In[66]:


hotel_df[["Review","Reviews_Lemm","Sentiment_VADER"]][hotel_df["Sentiment_VADER"] == hotel_df["Sentiment_VADER"].min()]


# In[67]:


DF_Neg_VADER = hotel_df[["Review","Reviews_Lemm","Sentiment_VADER"]][hotel_df["Sentiment_VADER"] < -0.1]


# In[68]:


DF_Neg_VADER 


# In[69]:


plt.figure(figsize = (9, 9))
wcloud_Neg_VADER = WordCloud(width = 900, height = 900, colormap = 'plasma', max_words = 150).generate(' '.join(pd.Series(DF_Neg_VADER["Review"].tolist()).astype(str)))
plt.imshow(wcloud_Neg_VADER)
plt.tight_layout(pad = 0.2)
plt.axis('off')
plt.show()


# # BI-GRAMS

# In[70]:


hotel_df["Bigrams"] = hotel_df["Reviews_Lemm"].apply(lambda x: list(ngrams(x, 2)))


# In[71]:


hotel_df


# In[72]:


bigrams_list = list(itertools.chain.from_iterable(hotel_df['Bigrams']))
bigrams_freq = collections.Counter(bigrams_list)

bigrams_freq_DF = pd.DataFrame(bigrams_freq.most_common(30), columns=['Bigrams', 'Count'])
bigrams_freq_DF


# In[73]:


plt.figure(figsize = (10,7))
sns.barplot(data = bigrams_freq_DF, x = "Count", y = "Bigrams")
plt.xlabel("Frequency")
plt.ylabel("Bigrams")
plt.title("30 Most Frequent Bigrams in Hard Rock Cafe NY Reviews")
plt.show()


# # TRI-GRAMS

# In[74]:


hotel_df["Trigrams"] = hotel_df["Reviews_Lemm"].apply(lambda x: list(ngrams(x, 3)))


# In[75]:


hotel_df


# In[76]:


trigrams_list = list(itertools.chain.from_iterable(hotel_df["Trigrams"]))
trigrams_freq = collections.Counter(trigrams_list)

trigrams_freq_DF = pd.DataFrame(trigrams_freq.most_common(40), columns=['Trigrams', 'Count'])
trigrams_freq_DF


# In[77]:


plt.figure(figsize = (10,7))
sns.barplot(data = trigrams_freq_DF, x = "Count", y = "Trigrams")
plt.xlabel("Frequency")
plt.ylabel("Trigrams")
plt.title("40 Most Frequent Trigrams in Hard Rock Cafe NY Reviews")
plt.show()


# In[78]:


hotel_df[["Sentiment_TextBlob", "Sentiment_VADER"]].describe()


# In[79]:


def sentiment_result(polarity):
    if polarity >= 0.1:
        return "Postive"
    elif polarity <= -0.3:
        return "Negative"
    else:
        return "Neutral"


# In[80]:


hotel_df["Label"] = hotel_df["Sentiment_VADER"].apply(lambda x: sentiment_result(x))


# In[81]:


hotel_df["Label"].value_counts()


# In[82]:


hotel_df


# In[90]:



data = hotel_df.to_csv(r'C:\\Users\\Hp\\Desktop\\Review.csv',index=True)

cwd


# In[85]:


data


# In[88]:


import os
cwd = os.getcwd()


# In[89]:


cwd


# In[91]:


import pandas as pd
import numpy as np
import os


# In[92]:


data = pd.read_csv("G:/Anaconda/envs/testenvs/Project-P39/Review.csv")


# In[93]:


print("columns name\n\n",data.columns)


# In[94]:


# dropping passed columns 
data.drop(['Unnamed: 0', 'Reviews_Token','Reviews_Lemm','Sentiment_TextBlob','Sentiment_VADER','Bigrams','Trigrams'], axis = 1, inplace = True) 


# In[95]:


data = pd.DataFrame(data) 


# In[96]:


data.head(3)


# In[97]:


data['Label'].unique
data['Label'] = data['Label'].replace({'Postive':0,'Neutral':1,'Negative':2})

# In[98]:


data.to_csv(r'model.csv',index=False)
os.getcwd()

