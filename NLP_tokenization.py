##takes list of titles or sentences, tokenizes them, gives a collocation and a regex 
##to find patterns in tokenized words.

import re
import pandas as pd
import numpy as np
import nltk
import itertools
import gensim
from nltk import word_tokenize
from nltk.text import TokenSearcher
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora, models, similarities

def remove_non_ascii(text):
	return ''.join(i for i in text if ord(i)<128)

titleSubs = []

df = pd.read_excel('derwent_patent.xlsx', skiprows = 1)
df.fillna('none', inplace=True)
df['Title'] = df['Title'].apply(remove_non_ascii)

titles = df[['Title']].as_matrix()
titleList = titles.tolist()
titlesAsList = list(itertools.chain.from_iterable(titleList)

tokenizedTitles = [word_tokenize(i) for i in titlesAsList]
titleTokens = list(itertools.chain.from_iterable(tokenizedTitles))
tokens = nltk.Text(titleTokens)
tokens.collocations()

i = 0
while i < len(tokenizedTitles):
	allVehic = TokenSearcher(nltk.Text(tokenizedTitles[i])).findall(r'<vehic.*|Vehic.*> <.*>')
	try:
		titleSubs.append([tokenizedTitles[i], allVehic[0]])
	except IndexError:
		titleSubs.append([tokenizedTitles[i], 'No Vehicle'])
	i += 1
	
columns = ['Broken Title', 'Subject']
dfNew = pd.DataFrame(titleSubs, columns = columns)
dfCombined = pd.DataFrame([df, dfNew], axis = 1)

writer = ExcelWriter('subject_assigned_data.xlsx')
dfCombined.to_excel(writer, 'combined data')
writer.save()

###############abstracts##############
stopwords = set(stopwords.words('english'))
stopwords.add('none') #for iterable, change to .update([item, item])

abstracts = df[['Abstract - DWPI']].as_matrix()
abstractList = abstracts.tolist()
abstractList = list(itertools.chain.from_iterable(abstractList))
texts = [[word for word in abstract.lower().split() if word not in stopwords] for abstract in abstractList]

frequency = defaultdict(int)
for text in texts:
	for token in text:
			frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('C:/Users/vkk/DerwentSample.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('C:/Users/vkk/DerwentCorpora.mm', corpus)

######TO DO: train a corpus on modeling topics #############
tfidf = models.TfidfModel(corpus) #initialize model out of corpus
doc_bow = [(0, 1), (1, 1)]

			
#tokenize without removing stopwords
#texts = [[abstract.lower().split() for abstract in abstracts] for abstracts in abstractList]