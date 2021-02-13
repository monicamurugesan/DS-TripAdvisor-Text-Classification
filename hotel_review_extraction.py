#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import csv
from selenium import webdriver


# In[2]:


path_to_file = "C:/Users/Hp/Hard-Rock-Reviews.csv"


# In[3]:


num_of_page =4521


# # Hotel_NAME
# # Hard_Rock_Cafe-New_York_City_New_York

# In[4]:


url = "https://www.tripadvisor.com/Restaurant_Review-g60763-d802686-Reviews-Hard_Rock_Cafe-New_York_City_New_York.html"


# In[5]:


if(len(sys.argv)==4):
    path_to_file = sys.argv[1]
    num_page = int(sys.argv[2])
    url = sys.argv[3]


# In[6]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager(version="87.0.4280.88").install())


# In[7]:


driver.get(url)


# In[8]:


# Open the file to save the review
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)

# In[9]:


import time
# change the value inside the range to save more or less reviews
for i in range(0, num_of_page):
    
    # expand the review 
    time.sleep(2)
    driver.find_element_by_xpath("//span[@class='taLnk ulBlueLinks']").click()

    container = driver.find_elements_by_xpath(".//div[@class='review-container']")

    for j in range(len(container)):

        title = container[j].find_element_by_xpath(".//span[@class='noQuotes']").text
        date = container[j].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
        rating = container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        review = container[j].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")

        csvWriter.writerow([date, rating, title, review]) 
        
    # change the page
    driver.find_element_by_xpath('.//a[@class="nav next ui_button primary"]').click()

driver.close()


# In[10]:


import pandas as pd
data = pd.read_csv("C:/Users/Hp/Hard-Rock-Reviews.csv")


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


print(len(data))


# # EDA

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize
from textblob import TextBlob

import wordcloud
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

from nltk.tokenize import word_tokenize
t=word_tokenize(data['Review'])
# In[15]:


data = pd.read_csv("C:/Users/Hp/Hard-Rock-Reviews.csv",encoding='latin-1')


# In[16]:


print(len(data))


# In[17]:


data.columns


# In[18]:


data.head()


# In[19]:


data.tail()


# In[20]:


dat1=data.rename(columns={'30 June 2020':'Date',
                    '40':'Rating',
                    'Always reliable':'Title',
                    'We ate at the bar after a show. Nachos and salad with a couple of glasses of wine. Just the right stuff at the right time in a energetic atmosphere. Thank you Duane!':'Review'},inplace = False)

dat1.to_csv()
# In[21]:


print(dat1.head())

path_to_file1=dat1.to_csv('Hard_Rock_New_Review.csv')
