#!/usr/bin/env python
# coding: utf-8

# In[1]:


responses = {
    "Hi": "Hello",
    "Whats your name?": "My name is Alice!",
    "How are you?": "I am good",
    "Bye": "See yaa!"
}


# In[2]:


def smalltalk():
    user = input("USER: ")
    if user in responses:
        print("BOT: {}\n".format(responses[user]))
        return smalltalk()
    else:
        print("BOT: Sorry, I am not that much developed to reply!\n") 
        return smalltalk()


# In[ ]:


smalltalk()


# In[ ]:




