# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:20:34 2019

@author: Prashant.Dhamale
"""
import pandas as pd
import scipy as sc
import numpy as np

name= ['Suraj', 'Atharva', 'Pranay']
Movie=['Avengers', 'Iron Man','Spiderman','Star Wars', 'Thor' ]
UserRatings=[['Suraj','Avengers',4],['Suraj','Star Wars', 2],['Suraj', 'Spiderman',5],['Suraj','Iron Man', 4],
             ['Atharva', 'Avengers', 5],['Atharva','Star Wars', 3],['Atharva', 'Thor', 5], ['Atharva', 'Iron Man', 3],
             ['Pranay', 'Avengers', 3],['Pranay', 'Thor', 4],['Pranay', 'Spiderman', 4],['Pranay', 'Iron Man', 3]]
dfUR= pd.DataFrame(UserRatings,columns = ['Name','Movie','Ratings'])

df2=dfUR.pivot(index = 'Name', columns='Movie', values='Ratings')   

ItemSim=np.zeros((5,5))
ItemSim1=pd.DataFrame(ItemSim,columns=['Avengers', 'Iron Man','Spiderman','Star Wars', 'Thor' ],
                      index=['Avengers', 'Iron Man','Spiderman','Star Wars', 'Thor' ])

df2=df2.fillna(0)
#df2['Avengers']['Atharva']
# Finding the Cosine Similarity Matrix
for x in Movie:
    for y in Movie:
        if(df2.isnull[y]):
        ItemSim1[x][y]=np.dot(df2[x],df2[y])/(np.linalg.norm(df2[x])*np.linalg.norm(df2[y]))

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(np.transpose(df2), np.transpose(df2)))
ItemSim1
for x in Movie:
    print(x)