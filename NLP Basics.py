#!/usr/bin/env python
# coding: utf-8

# ### Tokenization

# In[39]:


import nltk
from nltk.tokenize import word_tokenize


# In[40]:


nltk.download('punkt')
word_tokenize("Hi there, I am Aditya")


# ### Plotting word length

# In[41]:


import matplotlib.pyplot as plt


# In[42]:


plt.hist([1, 5, 5, 7, 7, 7, 9])


# In[43]:


words = word_tokenize("I am gonna plot the word length using an awesome tool!")


# In[44]:


word_length = [len(i) for i in words]


# In[45]:


plt.hist(word_length)


# ### Word Counts with Bag of Words

# In[46]:


from collections import Counter


# In[47]:


count = Counter(word_tokenize("Cat is in the box, the Cat likes the box, the box contains Cat"))


# In[48]:


count


# In[51]:


count.most_common(n=3)


# ### Text Pre Processing

# In[52]:


from nltk.corpus import stopwords
nltk.download('stopwords')


# In[53]:


text = "The cat is in the box. The cat likes the box. The box is over the cat."


# In[54]:


tokens = [word for word in word_tokenize(text.lower()) if word.isalpha()]


# In[55]:


tokens


# In[56]:


no_stops = [t for t in tokens
           if t not in stopwords.words('english')]


# In[57]:


no_stops


# In[58]:


Counter(no_stops).most_common(2)

