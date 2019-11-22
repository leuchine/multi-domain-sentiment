import os
import collections
import pickle
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import re


#insert words of a file
def insert_word(f):
	global all_words
	for l in f:
		words=re.split('\s|-',l.lower().split("|||")[0].strip())
		
		all_words+=words

#convert words to numbers
def convert_words_to_number(f, dataset, labels):
	global common_word
	for l in f:
		try:
			words=re.split('\s|-',l.lower().split("|||")[0].strip())
			label=l.lower().split("|||")[1].strip('\n')
			words=[common_word[w] if w in common_word else 1 for w in words]
			dataset+=[words]
			labels+=[label]
		except:
			continue
vocab=10000
gap=2
vocab_size=vocab-2
location='./dataset/'
all_words=[]

#iterate all files
for file in os.listdir(location):
	if file != '.DS_Store':
		with open(location+file+"/trn") as f:
			insert_word(f)
		with open(location+file+"/dev") as f:
			insert_word(f)

#take out frequent words 
counter=collections.Counter(all_words)
common_word=dict(counter.most_common(vocab_size))

#number them
c=2
for key in common_word:
	common_word[key]=c
	c+=1
print(common_word)
pickle.dump(common_word, open('dictionary', 'wb'))

for file in os.listdir(location):

	if file != '.DS_Store':
		train=[]
		train_label=[]
		test=[]
		test_label=[]
		with open(location+file+"/trn") as f:
			convert_words_to_number(f, train, train_label)

		with open(location+file+"/dev") as f:
			convert_words_to_number(f, train, train_label)

		pickle.dump(((train,train_label) ,(test,test_label)), open(location+file+'/dataset', 'wb'))


#create embedding vector matrix
word_vectors = KeyedVectors.load_word2vec_format('vectors.gz', binary=True)
word2vec=[[0]*300, [0]*300]
for number, word in sorted(zip(common_word.values(), common_word.keys())):
	try:
		print(type(word_vectors.word_vec(word)))
		word2vec.append(word_vectors.word_vec(word).tolist())
	except KeyError: 
		print(word+ " not found")
		word2vec.append([0]*300)
pickle.dump(word2vec, open('vectors', 'wb'))
print(len(word2vec))

print(word_vectors.word_vec('laptop'))
