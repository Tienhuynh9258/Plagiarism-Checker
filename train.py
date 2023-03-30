from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import os

def extractParas(text):
	"""
	Extract paras from the raw text provided

	Parameters:  
	str: raw text data from the input file  

	Returns:  
	list: List of paragraphs from the input document  
	"""

	paragraphs = []
	para = ""
	i = 0
	while i < len(text):
		if i < len(text) and text[i] == '\n':
			if para != "":
				paragraphs.append(para)
			para = ""
		else:
			para += text[i]
		i += 1
	if para != "\n" and para != "":
		paragraphs.append(para)

	return paragraphs

def tokenize(text):
	"""
	Tokenize the given paragraph

	Parameters:  
	list: List of paragraphs from the input document  

	Returns:  
	list: List of tokens from the input paragraph  
	"""

	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	#Convert to lower case
	for i, token in enumerate(tokens):
		tokens[i] = token.lower()
	return tokens

def stopwordRemoval(tokens):
	"""
	Remove common English stopwords from the tokens
	
	Parameters:  
	list: List of tokens  

	Returns:  
	list: List of tokens without the stop words  
	"""

	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	return tokens

def stemmer(tokens):
	"""
	Stemming function

	Parameters:  
	list: List of tokens without stopwords  

	Returns:  
	list: List of stemmed tokens  
	"""
	stemmer = PorterStemmer()
	for i, token in enumerate(tokens):
		tokens[i] = stemmer.stem(token)
	return tokens

if __name__ == '__main__':
	
	# GET THE FIRST DATA FROM FOLDER DATASET

	documents = []
	filename = {}
	vocab = set()
	# Data dictionary
	result = dict()
	files = os.listdir('DATASET')
	# Get list of documents that contains a list of paragraphs of a file
	# Get set of vocabulary  
	for i,file in enumerate(files):
		filename[i] = file
		with open('DATASET/' + file, encoding="utf8", errors='ignore') as f:
			data = f.read()
			paras = extractParas(data)
			paragraphs = []
			for j,para in enumerate(paras):
				# Preprocessing
				tokens = tokenize(para)
				stoplesstokens = stopwordRemoval(tokens)
				finaltokens = stemmer(stoplesstokens)
				
				paragraphs.append(finaltokens)
				for term in finaltokens:
					vocab.add(term)
			documents.append(paragraphs)
	
	result["documents"] = documents
	result["filenames"] = filename
	result["vocab"] = vocab

	# Add set to train file
	with open('train_vocab.txt', 'w', encoding="utf8", errors="ignore") as f:
		f.write(str(result))
	f.close()

	
	print("----------------------------Create dataset success------------------------------------------")
	