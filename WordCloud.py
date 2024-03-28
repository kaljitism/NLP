#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().system('pip install wordcloud')


# In[49]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[18]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


# In[11]:


with open('lyrics.txt') as txt:
    text = txt.read()


# In[12]:


text


# In[32]:


tokens = word_tokenize(text)
tokens_n = [word for word in tokens
         if word not in stopwords.words('english')]


# In[28]:


count_freq = Counter(tokens_n)


# In[44]:


most_freq = count_freq.most_common(20)
most_freq


# In[45]:


string = ""
for word, freq in most_freq:
    string = string + " " + word


# In[46]:


wordcloud_est = WordCloud(width=800, height=800,
                     background_color='white',
                     min_font_size=10)
wordcloud = wordcloud_est.generate(string)


# In[47]:


plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis=('off')
plt.tight_layout(pad = 0)


# In[ ]:




