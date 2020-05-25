#!/usr/bin/env python
# coding: utf-8

# Goal:
# Generate/find 10 documents ( e.g.‘.csv’, delimiter =‘,’)
# Write code in Ipython notebook counting the number of words in all documents
# Display 10 most popular words on a screen

# In[1]:

from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StringType
from pyspark import SQLContext
import pyspark.sql.functions as f
import pyspark.sql.types as T
from collections import Counter
import operator
import os

import nltk
from nltk.corpus import stopwords

from numpy import take

import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud



conf = (SparkConf()
    .setMaster("local")
    .setAppName("My app")
    .set("spark.executor.memory", "1g"))

sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


# ### Import of csv files

billdf = sqlContext.read.format('csv').options(header = 'true').load(r"C:\Users\Igor\YandexDisk\Документы\Univer\Seminar\data\*.csv")

billdf.show(10)

# ### Assigning data types to columns

billdf = billdf.withColumn('Rank2',billdf['Rank'].astype('integer'))
billdf = billdf.drop('Rank')
billdf = billdf.withColumnRenamed("Rank2", "Rank")

billdf = billdf.withColumn('Year2',billdf['Year'].astype('integer'))
billdf = billdf.drop('Year')
billdf = billdf.withColumnRenamed("Year2", "Year")

billdf = billdf.withColumn('Source2',billdf['Source'].astype('integer'))
billdf = billdf.drop('Source')
billdf = billdf.withColumnRenamed("Source2", "Source")


# ### Creating new dataframe for counting words

newdf = billdf.withColumn('word', f.explode(f.split(f.col('Lyrics'), ' '))).groupBy('word').count()
newdf = newdf.withColumnRenamed("count", "wcount")
mvv = newdf.sort('wcount', ascending=False).select("word").limit(10).rdd.flatMap(lambda x: x).collect()
#newdf.sort('wcount', ascending=False).show(10)

# ### Counting a number of words in the files

def count_words(billdf):
    
    words_count = 0

    for column in billdf.columns:
        temp_df = billdf.withColumn('word', f.explode(f.split(f.col(column), ' '))).groupBy('word').count().select(f.sum('count')).collect()[0][0]
    
        words_count += temp_df
        
    ret = int(words_count+len(billdf.columns))

    return ret

words = count_words(billdf)

# ## Results:

# #### The number of words in all documents:


print(f"Total amount of words is {words}")


# #### 10 most popular words:

print(f"Top 10 words {mvv}")

# Preparation for calculating stats

stpwr = set(stopwords.words('english'))
stpwr.update([' ','  ','   ','    ','','like','im','oh','dont','im'])

clean_df = newdf.filter(newdf['word'].isin(stpwr)==False)
#clean_df.sort(clean_df.wcount,ascending=False).show(10)

billdf = billdf.withColumn('allWords', f.size(f.split(f.col('Lyrics'), ' ')))

billdf = billdf.withColumn('uniqWords', f.split(f.col('Lyrics'), ' '))

billdf = billdf.withColumn('uniqWords',f.size(f.array_distinct("uniqWords")))
billdf = billdf.withColumn('Gini',f.col("uniqWords")/f.col("allWords"))

df2 = billdf.filter(billdf.allWords > 1)
#df2.show(10)

gini_df = df2.select(['Year','Gini']).groupby('Year').mean('Gini').withColumnRenamed('avg(Gini)', "Mean Gini")
#gini_df.orderBy('Year', ascending = True).show(20)

# Scatter plot of Gini coefficient

p_gini_df = gini_df.toPandas()


fig3 = px.scatter(p_gini_df, x='Year', y='Mean Gini', trendline="ols", title='Scatter plot of Gini coefficient by year')

fig3.show()


# Wordcloud of words in Lyrics
#%%
# Filter for stop-words contained in a "stpwr" set
udf_filter_words = f.udf(
    lambda x: [i for i in filter(None,x) if i not in stpwr])

# Counting words 
udf_flatten_counter = f.udf(
    lambda x: dict(Counter(x)),
    T.MapType(T.StringType(), T.IntegerType()))

def sort_dict_f(x):
    sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
    return sorted_x[:5]

# Schema for the user-defined function
schema = T.ArrayType(T.StructType([
    T.StructField("word", T.StringType(), False), T.StructField("count", T.IntegerType(), False)
]))

# Sorting dictionary in the ascending order
SorterUDF = f.udf(sort_dict_f, schema)

udf_take_n_words = f.udf(
    lambda x: [i for i in x[:5]])

# Stopwords filter is applied
testdf =  billdf.withColumn('lst', f.split(f.col('Lyrics'), ' '))
testdf2 = testdf.select(['Year','lst']).groupby('Year').agg(f.collect_list('lst'))\
                                        .withColumn("collect_list(lst)",f.flatten("collect_list(lst)"))\
                                        .withColumnRenamed("collect_list(lst)", "All words")
testdf3 = testdf2.withColumn("cnt", SorterUDF(udf_flatten_counter(udf_filter_words("All words"))))
lsc = testdf3.select("All words").collect()
lsc = [i for i in lsc[0][0] if i not in stpwr]
ds = dict(Counter(lsc))
ds = sorted(ds.items(), key=operator.itemgetter(1),reverse=True)

popular_words = pd.Series(lsc).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='white').generate(popular_words)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#sc.stop()

