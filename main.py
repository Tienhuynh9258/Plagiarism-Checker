import nltk
from train import extractParas, tokenize, stopwordRemoval, stemmer
import sys
import math
import numpy as np
import time
import ast

"""
---------------------------------------CONSIN-SIMILARITY----------------------------------------------
"""

# Use for cosine similarity
def createInvertedIndex(documents):
	"""
	Creating the inverted index

	Parameters:  
	documents(list): Preprocessed list containing content of the documents  

	Returns:  
	Dictionary: The inverted index  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: posting list  
	"""

	inverted_index = {}
	for i,doc in enumerate(documents):
		for j, para in enumerate(doc):
			for word in para:
				# print(word)
				if inverted_index.get(word,False):
					if i not in inverted_index[word]:
						inverted_index[word].append(i)
				else:
					inverted_index[word] = [i]
	return inverted_index

# Use for cosine similarity
def calculate_TfIdf_Weights(vocab, inverted_index, documents):
	"""
	Calculate the tfidf scores 

	Parameters:  
	(arg1)vocab(set): Vocabulary of training data  
	(arg2)inverted_index(dict): Inverted Index  
	(arg3)documents(list): Preprocessed list containing content of the documents  


	Returns:  
	Dictionary: TfIdf scores  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: list of tf * idf scores for all the documents  
	"""
	
	tf_idf = {}
	count = 0
	for word in vocab:
		for i,doc in enumerate(documents):
			if i in inverted_index[word]:
				for para in doc:
					for w in para:
						if(w == word):
							count+=1
			else:
				count = 0

			if (i != 0):
				tf_idf[word].append(count)
			else:
				tf_idf[word] = [count]
			count = 0

	#convert to 1 + log(tf)
	for word in vocab:
		tf_idf[word] = [(1 + math.log(x+1)) for x in tf_idf[word]]

	#add idf weighting
	totaldocs = len(documents)
	for word in vocab:
		idfval = math.log(totaldocs/len(inverted_index[word]))
		tf_idf[word] = [round(x * idfval, 3) for x in tf_idf[word]]

	return tf_idf

# Use for cosine similarity
def calculateQueryWeights(testvocab, inverted_index, testparagraphs, totaldocs):
	"""
	Calculate weight of the query passed - here query is a document 

	Parameters:  
	arg1: testvocab(set): Preprocessed list containing content of the documents  
	arg2: inverted_index(dict): Inverted Index  
	arg3: testparagraphs(list): paragraphs extracted from test document  
	arg4: totaldocs(int): Number of documents in Training corpus  

	Returns:  
	weights(dictionary)  
	&nbsp;&nbsp; Key: word in test document vocabulary  
	&nbsp;&nbsp; Value: TfIdf score  
	"""
	weights = {}
	
	for word in testvocab:
		count = 0
		for para in testparagraphs:
			for w in para:
				if(w == word):
					count += 1
		weights[word] = count

	for word in testvocab:
		weights[word] = 1 + math.log(weights[word]+1)

	
	for word in testvocab:
		if word not in inverted_index:
			inverted_index[word] = []
			tf_idf[word] = [0]*totaldocs
		idfval = math.log((totaldocs+1)/(len(inverted_index[word])+1))
		
		weights[word] = round(weights[word] * idfval, 3)

	return weights

# Use for cosine similarity
def calculateParaWeights(para, vocab, inverted_index, totaldocs):
	"""
	Calculate weight of the query passed - here query is a paragraph 

	Parameters:  
	arg1: para(list): List of tokens in the query paragraph  
	arg2: testvocab(set): Preprocessed list containing content of the query paragraph  
	arg3: inverted_index(dict): Inverted Index  
	arg4: totaldocs(int): Number of documents 

	Returns:  
	weights(numpy array) - contains weights of the words in the test para vocabulary for the query paragraph  
	"""

	weights = np.zeros(len(vocab))
	for i,word in enumerate(vocab):
		count = 0
		for w in para:
			if(w == word):
				count += 1
		weights[i] = count

		for i in range(len(vocab)):
			if(weights[i] != 0):
				weights[i] = 1 + math.log(weights[i]+1)

		for i,word in enumerate(vocab):
			if word not in inverted_index:
				inverted_index[word] = []
				tf_idf[word] = [0]*totaldocs
			idfval = math.log((totaldocs+1)/(len(inverted_index[word])+1))
			weights[i] = round(weights[i] * idfval, 3)
	return weights

# Use for cosine similarity
def rankDocsByCosineSimilarity(documents, testparagraphs, tf_idf, inverted_index):
	"""
	Calculate the ranking of the docs wrt cosine similarity to testdoc 

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: testparagraphs(list): paragraphs extracted from test document  
	arg3: tf_idf(dict): TfIdf scores  
	&nbsp;&nbsp; Key: word  
	&nbsp;&nbsp; Value: list of tf * idf scores for all the documents  
	arg4: inverted_index(dict): Inverted Index  

	Returns:  
	Dictionary: Ranking  
	&nbsp;&nbsp; Key: doc number  
	&nbsp;&nbsp; Value: Cosine Similarity score  
	"""
	ranking = {}
	testvocab = set()
	for para in testparagraphs:
		for word in para:
			testvocab.add(word)
	testdoc = []
	testdoc.append(testparagraphs)
	query_tfidf = calculateQueryWeights(testvocab, inverted_index, testparagraphs, len(documents))
	a = np.zeros(len(testvocab))
	b = np.zeros(len(testvocab))
	
	for j,doc in enumerate(documents):
		i = 0
		for word in testvocab:
			if word in inverted_index and j in inverted_index[word]:
				a[i] = tf_idf[word][j]
			else:
				a[i] = 0
			b[i] = query_tfidf[word]
			i += 1
		ranking[j] = cosine_sim(a,b)

	#sort ranking dict in reverse order by keys
	ranking = {k: v for k, v in sorted(ranking.items(),reverse = True, key=lambda item: item[1])}
	return ranking

# Use for cosine similarity
def cosine_sim(a,b):
	"""
	Calculate cosine of two vectors
	
	Parameters:  
	arg1: a(list) - 1st vector
	arg2: b(list) - 2nd vector

	Return:
	cos_sim(float) - Computed cosine
	"""

	cos_sim = 0
	if np.linalg.norm(a)!=0 and np.linalg.norm(b)!=0:
		cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
	return cos_sim

"""
---------------------------------------JACCARD-SIMILARITY----------------------------------------------
"""

# Use for jaccard distance similarity
def rankDocsByJaccardSimilarity(documents, testparagraphs):
	"""
	Calculate the ranking of the docs by jaccard similarity to testdoc 

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: testparagraphs(list): paragraphs extracted from test document   

	Returns:  
	Dictionary: Ranking  
	&nbsp;&nbsp; Key: doc number  
	&nbsp;&nbsp; Value: Jaccard Similarity score  
	"""

	# Get list of words in test document
	tokens_test = [word for para in testparagraphs for word in para] 
	#----------------
	ranking = {}
	
	for j,doc in enumerate(documents):
		tokens_document = [word for para in doc for word in para]
		document_chars = set(nltk.ngrams(tokens_document, n=3))
		test_chars = set(nltk.ngrams(tokens_test, n=3))

		# Subtracting of 1- nltk.jaccard_distance towards to 0 means it is more different, otherwise towards 1 means more similar
		ranking[j] = 1 - nltk.jaccard_distance(document_chars, test_chars) if document_chars & test_chars else 0

	# Sort ranking dict in reverse order by keys
	ranking = {k: v for k, v in sorted(ranking.items(),reverse = True, key=lambda item: item[1])}
	return ranking

# Use for jaccard distance similarity
def calculateParasByJaccardSimilarity(documents, paragraph):
	"""
	Calculate the best similar paragraph of document by jaccard similarity to testpara

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: paragraph(list): content of a paragraph  

	Returns:  
	a tuple of paragraph and document containing it if the highest jaccard score of a paragraph in documents is larger than 0.95, else empty tuple
	"""

	#----------------
	ranking = {}
	highest_score = 0
	highest_document = 0
	highest_paragrpaph = 0
	
	# Create a set that contains list of jaccard score for paragraphs for each document
	for i,doc in enumerate(documents):
		scoreInDoc = list()
		for j,para in enumerate(doc):
			document_chars = set(nltk.ngrams(para, n=3))
			test_chars = set(nltk.ngrams(paragraph, n=3))

			scoreInDoc.append(1-nltk.jaccard_distance(document_chars, test_chars) if document_chars and test_chars else 0)  
		ranking[i] = scoreInDoc

	# Get the highest jaccard score
	for i in ranking:
		values = ranking[i]
		for j,score in enumerate(values):
			if score > highest_score:
				highest_score = score
				highest_paragrpaph = j
				highest_document = i

	if highest_score > 0.95:
		return tuple([highest_document, highest_paragrpaph])
		
	return tuple()

"""
---------------------------------------EDIT-DISTANCE-SIMILARITY----------------------------------------------
"""

# Use for edit distance similarity
def rankDocsByEditDistanceSimilarity(documents, testparagraphs):
	"""
	Calculate the ranking of the docs by edit distance similarity to testdoc 

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: testparagraphs(list): paragraphs extracted from test document   

	Returns:  
	Dictionary: Ranking  
	&nbsp;&nbsp; Key: doc number  
	&nbsp;&nbsp; Value: Edit distance Similarity score  
	"""

	# Get list of words in test document
	tokens_test = [word for para in testparagraphs for word in para] 
	#----------------
	ranking = {}
	
	for j,doc in enumerate(documents):
		tokens_document = [word for para in doc for word in para]
        
		# The number of operations need to do for the source text similaring the target text are smaller that means they are more similar
		ranking[j] = nltk.edit_distance(tokens_document, tokens_test)

	# Sort ranking dict in order by keys
	ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1])}
	return ranking

# Use for edit distance similarity
def calculateParasByEditDistanceSimilarity(documents, paragraph):
	"""
	Calculate the best similar paragraph of document by edit distance similarity to testpara

	Parameters:  
	arg1: documents(list): Preprocessed list containing content of the documents  
	arg2: paragraph(list): content of a paragraph  

	Returns:  
	a tuple of paragraph and document containing it if the smallest edit distance score of a paragraph in documents is smaller than 10, else empty tuple
	"""

	#----------------
	ranking = {}
	smallest_score = 1000
	smallest_document = 0
	smallest_paragrpaph = 0
	
	# Create a set that contains list of edit distance score for paragraphs for each document
	for i,doc in enumerate(documents):
		scoreInDoc = list()
		for j,para in enumerate(doc):

			scoreInDoc.append(nltk.edit_distance(para, paragraph) if para and paragraph else 1000)  
		ranking[i] = scoreInDoc

	# Get the smallest distance score
	for i in ranking:
		values = ranking[i]
		for j,score in enumerate(values):
			if score < smallest_score:
				smallest_score = score
				smallest_paragrpaph = j
				smallest_document = i

	if smallest_score < 10:
		return tuple([smallest_document, smallest_paragrpaph])
		
	return tuple()

if __name__ == '__main__':
	start = time.time()

	# Load set from file
	my_set = set()
	with open('train_vocab.txt', 'r', encoding="utf8", errors="ignore") as f:
		my_set = ast.literal_eval(f.read())
	f.close()
	documents = my_set.get('documents')
	filenames = my_set.get('filenames')
	vocab = my_set.get('vocab')
	

	# Take test testfile name from cl arguments
	testDocument = str(sys.argv[1])

	with open(testDocument, encoding="utf8", errors="ignore") as input_file:
		testdata = input_file.read()

	# Process test document
	paras = extractParas(testdata)
	testparagraphs = []
	for j, para in enumerate(paras):
		# Preprocessing
		tokens = tokenize(para)
		stoplesstokens = stopwordRemoval(tokens)
		finaltokens = stemmer(stoplesstokens)
		
		testparagraphs.append(finaltokens)
	
	if sys.argv[2] == 'Jaccard':
		# Ranking documents from the highest similarity to smallest
		ranking = rankDocsByJaccardSimilarity(documents, testparagraphs)

		# Calculate the similarity of each paragraph in test document with the paragraphs of dataset documents
		# Only print when the similarity value is larger than the defined value 
		print("\n")
		print("Calculating document uniqueness...")
		
		totalmatches = 0
		for k, tpara in enumerate(testparagraphs):
			response = calculateParasByJaccardSimilarity(documents, tpara)
			if response:
				totalmatches += 1
				print("Paragraph ",k," from test document matches with paragraph ",response[1]," from document ",response[0]," - ",filenames[response[0]])

		print("Document uniqueness = ", round(100 - (totalmatches*100/len(testparagraphs)), 2),"%")
		
	elif sys.argv[2] == 'Cosine':
		# Process inverted index and calculate tf_idf scores
		inverted_index = createInvertedIndex(documents)	
		
		tf_idf = calculate_TfIdf_Weights(vocab, inverted_index, documents)

		# Ranking documents from the highest similarity to smallest
		ranking = rankDocsByCosineSimilarity(documents, testparagraphs, tf_idf, inverted_index)

		# Calculate the similarity of each paragraph in test document with the paragraphs of dataset documents
		# Only print when the similarity value is larger than the defined value 
		print("\n")
		print("Calculating document uniqueness...")
		
		totaldocs = len(documents)
		totalmatches = 0
		for k, tpara in enumerate(testparagraphs):
			testparavocab = set()
			for term in tpara:
				testparavocab.add(term)
			testparaweights = calculateParaWeights(tpara, testparavocab, inverted_index, totaldocs)
			matchfound = 0
			for i,doc in enumerate(documents):
				for j,para in enumerate(doc):
					paraweights = calculateParaWeights(para, testparavocab, inverted_index, totaldocs)
					cos_sim = cosine_sim(testparaweights, paraweights)
					if(cos_sim > 0.95):
						matchfound = 1
						totalmatches += 1
						print("Paragraph ",k," from test document matches with paragraph ",j," from document ",i," - ",filenames[i])
						break
				if(matchfound == 1):
					break
		print("Document uniqueness = ", round(100 - (totalmatches*100/len(testparagraphs)), 2),"%")

	else: # 'Edit'
		# Ranking documents from the highest similarity to smallest
		ranking = rankDocsByEditDistanceSimilarity(documents, testparagraphs)

		# Calculate the similarity of each paragraph in test document with the paragraphs of dataset documents
		# Only print when the similarity value is smaller than the defined value 
		print("\n")
		print("Calculating document uniqueness...")
		
		totalmatches = 0
		for k, tpara in enumerate(testparagraphs):
			response = calculateParasByEditDistanceSimilarity(documents, tpara)
			if response:
				totalmatches += 1
				print("Paragraph ",k," from test document matches with paragraph ",response[1]," from document ",response[0]," - ",filenames[response[0]])

		print("Document uniqueness = ", round(100 - (totalmatches*100/len(testparagraphs)), 2),"%")

	# Print top 10 matching documents
	print("\nTop 10 documents matching the given test document in ranked order are: ")
	print("Rank","	-	","Doc No", "	-	","Doc Name","		-	", "Cosine Scores" if sys.argv[2] == 'Cosine' else "Jaccard Scores" if sys.argv[2] == 'Jaccard' else "Edit distance Scores")
	cnt = 0
	for i in ranking:
		cnt += 1
		print(cnt,"	-	",i, "		-	",filenames[i],"	-	",ranking[i])
		if(cnt>9):
			break
	print("Total time taken = ", time.time() - start, "s")
