#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')

print(movies.info())


# In[4]:


print(movies.head())


# In[5]:


movies['overview'] = movies['overview'].fillna('')
print(movies['genres'])


# In[6]:


#converting JSON to a list of strings
import ast 
movies['genres'] = movies['genres'].apply ( lambda x: [d['name'] for d in ast.literal_eval(x)])

movies['genres'] = movies['genres'].apply( lambda x: ' '.join(x))
print(movies['genres'])


# In[7]:


movies['combined_features'] = movies['overview'] + ' ' + movies['genres']
print(movies['combined_features'])


# In[8]:


#distribution of movie genres

movies['genres'] = movies['genres'].str.split(' ')
all_genres = [genre for sublist in movies['genres'] for genre in sublist]
plt.figure(figsize = (12,8))
plt.hist(all_genres, bins=20, color = 'green')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.title('Distibution of Movie Genres')
plt.xticks(rotation = 90)
plt.show()


# In[9]:


# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
print(tfidf_matrix)


# In[10]:


# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


# In[11]:


#Series for movie title with index
indices = pd.Series(movies.index, index = movies['title']).drop_duplicates()
indices


# In[12]:


def get_recommendations(title, cosine_sim = cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        print(movie_indices)
        return movies['title'].iloc[movie_indices]

print(get_recommendations('Avatar'))


# In[ ]:





# In[ ]:




