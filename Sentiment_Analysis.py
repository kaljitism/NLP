#!/usr/bin/env python
# coding: utf-8

# ### Sentiment Analysis or Opinion Mining
# 
# Process of understanding the opinion of author about the subject
# 
# #### Sentiment Analysis System
# 
# 1. Opinion or Polarity - Positive, Neutral or Negative
# 2. Emotion - Joy, Surprise, Anger, Disgust
# 3. Subject of Discussion - What is being talked about
# 4. Opinion Holder or Entity - By whom?
# 
# #### Why Sentiment Analysis
# 
# 1. Social Media Monitoring - Not only what people are talking, but how they are talking about it in forums, post, blogs and news.
# 2. Brand Monitoring
# 3. Customer Service
# 4. Product Analysis
# 5. Market Research and Analysis

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('IMDB_sample.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.drop('Unnamed: 0', axis=1, inplace=True)


# In[6]:


data.head()


# In[7]:


data.label.value_counts()


# In[8]:


data.label.value_counts() / len(data)


# In[9]:


length_reviews = data.review.str.len()


# In[10]:


max(length_reviews)


# In[11]:


min(length_reviews)


# #### Level of Granularity(level of detail)
# 
# 1. Document Level(Blog, News)
# 2. Sentence Level(Sentence, Small Feedback)
# 3. Aspect Level(Opinions about multiple features)
# 
# #### Algorithms for Sentiment Analysis
# 
# 1. Rule/ Lexicon Based - Contains a list of words with Balanced score
# 2. Automated Systems based on Machine Learning - A classification problem
# 
# #### Valance - 
# A score
# 
# #### Subjectivity -
# The quality of being based on or influenced by personal feelings, tastes, or opinions. Measured from 0 to 1. 0 being Objective, while 1 being Subjective.
# 
# #### Polarity - 
# Measured between -1 and 1. -1 being very negative, 0 being neutral and 1 being positive.

# In[12]:


get_ipython().system('pip install textblob')


# In[13]:


from textblob import TextBlob


# In[14]:


my_valance = TextBlob("Today was a good day!")


# In[15]:


my_valance.polarity


# In[16]:


my_valance.subjectivity


# In[17]:


my_valance.pos_tags


# In[18]:


my_valance.detect_language()


# In[19]:


my_valance.sentiment


# ### Automated Machine Learning Approach
# 1. Rely on labelled Historical data.
# 2. Might take a while to train
# 3. Latest Machine Learning Models can be quite powerful.
# 
# 
# ### Rule/ Lexicon Based Approach
# 1. Rely on Manually crafted valance score
# 2. Different words might have different polarity in different context, which may make predictions opposite to what it actually is.
# 3. Can be quite fast.

# #### Word Cloud Algorithm
# Pictorial Representation of Words where the size is based on frequency of words. More frequent a word is, bigger and bolder it will appear on the word cloud.
# 
# ##### Pros 
# 1. Can reveal the essential
# 2. Provide an overall sense of the text.
# 3. Easy to grasp and engaging.
# 
# ##### Cons
# 1. Sometimes confusing and uninformative.
# 2. With larger text, require more work.

# In[20]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[21]:


two_cities = "It was the best of times, it was the worst of times,\
it was the age of wisdom, it was the age of foolishness,\
it was the epoch of belief, it was the epoch of incredulity,\
it was the season of Light, it was the season of Darkness,\
it was the spring of hope, it was the winter of despair,\
we had everything before us, we had nothing before us,\
we were all going direct to Heaven, we were all going,\
direct the other way - in short, the period was so far\
like the present period, that some of its noisiest\
authorities insisted on its being recieved, for good\
or for evil, in the superlative degree of comparison only."


# In[22]:


cloud_two_cities = WordCloud().generate(two_cities)


# In[23]:


cloud_two_cities


# In[24]:


plt.imshow(cloud_two_cities, interpolation='bilinear')
plt.axis('off')


# In[25]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[26]:


review_array = data.review.values
new_list = []
for i in review_array:
    new_list.append(i)
    
text = ""
for string in new_list:
    text += " " + string


# In[27]:


text[:100]


# In[28]:


tokens = [token for token in word_tokenize(text.lower()) if token.isalpha()]


# In[29]:


tokens[:10]


# In[30]:


new_data = [word for word in tokens if word not in stopwords.words('english')]


# In[31]:


new_data[:10]


# In[32]:


text = ""
for word in new_data:
    text += " " + word


# In[33]:


text[:100]


# In[34]:


cloud_imdb = WordCloud().generate(text)


# In[35]:


plt.imshow(cloud_imdb)
plt.axis('off')


# ### Bag of Words  
# 
# 1. Describes the occurence of words within a document or a collection of documents(corpus)
# 2. Builds a vocabulary of the words and measure of their presence.

# In[36]:


amazon = pd.read_csv('amazon_reviews_sample.csv')


# In[37]:


amazon.head()


# In[38]:


amazon.drop('Unnamed: 0', axis=1, inplace=True)


# In[39]:


from sklearn.feature_extraction.text import CountVectorizer


# In[40]:


vect = CountVectorizer(max_features=1000)
vect.fit(amazon.review)
X = vect.transform(amazon.review)


# In[41]:


X


# In[42]:


import numpy as np


# In[43]:


my_array = X.toarray()


# In[44]:


my_array


# In[45]:


X_df = pd.DataFrame(my_array, columns=vect.get_feature_names())


# In[46]:


X_df.head()


# #### Remember that context matters!
# 
# *I am happy, not sad*
# 
# *I am sad, not happy*
# 
# Putting 'not' in front of word(negation) is one example of how context matters.
# Bag of Words will be same for both of the sentences.
# 
# ### Capturing Context with Bag of Words
# 
# 1. Unigrams - single tokens
# 2. Bigrams - pairs of tokens
# 3. Trigrams - triplets of tokens
# 4. n-grams - sequence of n-tokens
# 
# For example,
# *The weather today is wonderful*
# 
# 1. Unigrams: {The, weather, today, is, wonderful}
# 2. Bigrams: {The weather, weather today, today is, is wonderful}
# 3. Trigrams: {The weather today, weather today is, today is wonderful}

# In[47]:


vect = CountVectorizer(ngram_range=(1, 2))
vect.fit(amazon.review)
X_r = vect.transform(amazon.review)


# In[48]:


X_review = pd.DataFrame(X_r.toarray(), columns=vect.get_feature_names())


# In[49]:


vect = CountVectorizer(max_features=1000, ngram_range=(2, 2), max_df=500)
vect.fit(amazon.review)
X_review = vect.transform(amazon.review)


# In[50]:


X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())


# In[51]:


X_df.head()


# In[52]:


word_tokens = [word_tokenize(review) for review in amazon.review]


# In[53]:


len_tokens = []
for i in range(len(word_tokens)):
    len_tokens.append(len(word_tokens[i]))


# In[54]:


amazon['n_tokens'] = len_tokens


# In[55]:


amazon.head()


# In[56]:


get_ipython().system('pip install langdetect')


# In[57]:


from langdetect import detect_langs


# In[58]:


foreign = "Este libro ha sido uno de los mehores libros que he leido"


# In[59]:


detect_langs(foreign)


# In[61]:


languages = []
for row in range(len(amazon)):
    languages.append(detect_langs(amazon.iloc[row, 1]))


# In[62]:


languages = [str(lang).split(':')[0][1:] for lang in languages]


# In[63]:


amazon['Languages'] = languages


# In[64]:


amazon.head()


# In[65]:


from wordcloud import WordCloud, STOPWORDS


# In[66]:


my_stopwords = set(STOPWORDS)
my_stopwords.update(["movie", "movies", "film", "films", "watch", "see", "think", "br", "time", "one", "character", "story", "show", "made", "make", "even"])


# In[67]:


my_cloud = WordCloud(stopwords=my_stopwords).generate(text)
plt.imshow(my_cloud, interpolation='bilinear')


# In[68]:


from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


# In[69]:


my_stop_words = ENGLISH_STOP_WORDS.union(['film', 'movie', 'cinema', 'theatre'])


# In[70]:


vect = CountVectorizer(stop_words=my_stop_words)
vect.fit(amazon.review)
X = vect.transform(amazon.review)


# #### Stemming
# 
# Stemming is the process of transforming words to their root forms, even if the stem itself is not a valid word in the language.
# 
# 1. Produces roots of words
# 2. Fast and effecient to compute
# 
# #### Lemmatization
# 
# Lemmatization is quite similar to stemming but unlike, it reduces the words to roots that are valid words in the language.
# 
# 1. Produces actual words
# 2. Slower than stemming and can depend on the part-of-speech

# In[71]:


from nltk.stem import PorterStemmer
porter = PorterStemmer()


# Snowball stemmer is used for Foreign Languages
# 
# *from nltk.stem.snowball import SnowballStemmer*
# 
# *DutchStemmer = SnowballStemmer("dutch")*
# 
# *DutchStemmer.stem("beginen")*
# 
# Outputs:
# *Begin*

# In[72]:


porter.stem("wonderful")


# In[73]:


porter.stem("Today is a wonderful day!")


# In[74]:


stemmed = [porter.stem(word) for word in word_tokenize("Today is a wonderful day!")]


# In[75]:


stemmed


# In[76]:


import nltk
from nltk.stem import WordNetLemmatizer
WNLemmatizer = WordNetLemmatizer()
nltk.download('wordnet')


# In[77]:


WNLemmatizer.lemmatize('wonderful', pos='a')


# ### TF-IDF
# 
# #### TF: Term Frequency:
# How often a given word appears within a document in the corpus
# 
# #### IDF: Inverse Document Frequency:
# Log-ratio between the total number of documents and the 
# number of documents that contain a specific word
# 
# Used to calculate the weight of words that do not occur frequently.
# 
# #### TFIDF Score:
# *tfidf = term frequency * inverse document frequency*
# 
# 1. *BOW does not account for length of a document, tfidf does*
# 2. *tfidf likely to capture words common within a document but not across documents*
# 3. *Since it penalizes the frequent words, less need to deal with stop words explicitly*
# 4. *Quite useful in search queries and information retrieval to rank the relevance of returned results*

# In[78]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[79]:


vect = TfidfVectorizer(max_features=100).fit(amazon.review)


# In[80]:


X = vect.transform(amazon.review)


# In[81]:


X_tfidf = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
X_tfidf.head()


# In[82]:


from sklearn.linear_model import LogisticRegression


# In[83]:


vect = TfidfVectorizer(max_features=100).fit(data.review)
X_imdb_tfidf = vect.transform(data.review)
X_imdb = pd.DataFrame(X_imdb_tfidf.toarray(), columns=vect.get_feature_names())


# In[84]:


X, y = X_imdb, data.label.values


# In[85]:


log_reg = LogisticRegression().fit(X, y)


# In[86]:


log_reg.score(X, y)

