#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[7]:


def greet():
    user = input("USER: ")
    user = user.lower()
    match = re.search(r"(hi|hello|hey)", user)
    if match:
        return "Hey!"


# In[8]:


greet()


# In[ ]:




