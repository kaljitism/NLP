#!/usr/bin/env python
# coding: utf-8

# ### Creating a Dictionary and Corpus using Gensim

# In[2]:


get_ipython().system('pip install gensim')


# In[5]:


from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize


# In[6]:


docs = ['Hello Everyone, I am creating Dictionary',
       'How are you all. Hows the work',
       'I am not liking my time mismanagement',
       'I have to work harder',
       'Its me, ME, ME, ME']


# In[9]:


tokenized_docs = [word_tokenize(doc.lower()) for doc in docs]


# In[10]:


tokenized_docs


# In[11]:


dictionary = Dictionary(tokenized_docs)


# In[14]:


print(dictionary)


# In[15]:


dictionary.token2id


# In[16]:


corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]


# In[17]:


corpus


# ### Tf - idf
# Term frequency - inverse document frequency

# In[18]:


from gensim.models.tfidfmodel import TfidfModel


# In[19]:


tfidf = TfidfModel(corpus)


# In[22]:


tfidf[corpus[0]]

