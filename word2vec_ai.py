import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import gensim
import nltk
import utils2

from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.text import TokenSearcher
	
def tsne_plot_word(model, word):
    #Creates TSNE model of top 20 most similar words and plots it
	
	close_words = model.most_similar(positive=[word], negative = [], topn = 20)
	
    labels = [word]
    tokens = [model[word[0]]]

    for new_word in close_words:
        tokens.append(model[new_word[0]])
        labels.append(new_word[0])
    
	tsne = TSNE(n_components=2, random_state=0)
	Y = tsne.fit_transform(tokens)
	
	x_coords = Y[:, 0]
	y_coords = Y[:, 1]
	
	plt.scatter(x_coords, y_coords)
	
	for label, x, y in zip(labels, x_coords, y_coords):
		plt.annotate(label, xy=(x, y), xytext=(0,0), textcoords='offset points')
	plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
	plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
	plt.show()

def tsne_plot_all(model):
	#Creates TSNE model of entire model - good practice to make model with small vocab
	
	labels = []
	tokens = []
	for word in model.wv.vocab:
		tokens.append(model[word])
		labels.append(word)
	tsne = TSNE(n_components=2, random_state=0)
	Y = tsne.fit_transform(tokens)
	x_coords = Y[:, 0]
	y_coords = Y[:, 1]
	plt.scatter(x_coords, y_coords)
	for label, x, y in zip(labels, x_coords, y_coords):
		plt.annotate(label, xy=(x, y), xytext = (0, 0), textcoords = 'offset points')
	plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
	plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
	plt.show()

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stopwords.update(['none', '<period>', '<comma>', '<question_mark>', 'quotation_mark', 
                   '<semicolon>', '<exclamation_mark>', '<hyphens>', '<left_paren>',
				   '<right_paren>', '<colon>', 'elsevier', 'ltd', 'rights', 'reserved', 
				   '2013', '2014', '2011', '2012', '2010', 'also', 'however', 'may', 'ieee', 
				   'use', 'using', 'used', '©'])

dfCols = ['abstract', 'title', 'cites', 'eid']
df = pd.read_csv('ai_doc_all_cites.csv', names = dfCols)
abs = df[['abstract']].as_matrix()
abs = abs.tolist()
abs = list(itertools.chain.from_iterable(abs))

absCl = []
i = 0
while i < len(abs):
	absCl.append(utils2.preprocess(abs[i]))
	i += 1

texts = [[word for word in abstract.lower().split() if word not in stopwords] for abstract in absCl]

frequency = defaultdict(int)
for text in texts:
	for token in text:
			frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]


#Word2Vec calculate most similar words, probability of text under model
model = gensim.models.Word2Vec(texts, min_count = 1, hs=1, negative = 0)
model.save('aimodel')
model.most_similar(positive=['neural'], negative = [], topn = 10)
model.score(['deep neural networks'.split()])

#NLTK TokenSearcher to ID abstracts containing desired words; change the regex text after findall according to 
#what you want to search for. Try using the most_similar function or the tsne_plot_word function to identify 
#words in a group
#this creates a list of two lists, the abstract tokens you created above and the words found using finall

y = []
i = 0
while i < len(texts):
	nnCite = TokenSearcher(nltk.Text(texts[i])).findall(r'<vsm|artificial|neural|network|supervised|bee|fgs-pid|lgrnn|anfis|feed-forward|hopfield|feedforward|bayes|bayes.*|machine|unsupervis.*|cluster|trees|learning|regression|backpropagation|algorithm.|algorithm|pso|rbf|fuzzy|linear|pattern|vector|vector.|function.|function|ai|ann|bsnn|rbfn|tdrnn|lvq|parameterization|imc|pls|nn|svm.*> <.*>')
	try:
		y.append([texts[i], nnCite[0]])
	except IndexError:
		y.append([texts[i], 'nope'])
	i += 1

#converts the list y into small dataframe, merges with original dataframe
#NOTE: remove undesired columns from original df (scopus provides a lot of blank garbage)

columns = ['abs', 'subject maybe']
dfNew = pd.DataFrame(y, columns = columns)
dfCombo = pd.concat([df, dfNew], axis = 1)
dfCombo.to_csv('subject_cites3.csv', encoding='utf-8', index=False)
dfSec = dfCombo[(dfCombo['subject maybe']== 'nope')]

#create a small model so you can use tsne_plot_all, checks length; adjust min_count to plot OK

smallmodel = gensim.models.Word2Vec(texts, size=100, window = 20, min_count = 150, workers = 4)
len(smallmodel.wv.vocab)