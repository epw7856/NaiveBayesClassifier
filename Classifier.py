# Naive Bayes Text Classifier
# Author: Eric Walker

###########################################################################################################
# Import libraries, define static variables, outline label dictionary mappings
###########################################################################################################

import os
import string
import re
import math
import time
import datetime
import sys
import csv

# Define "static" variables, stop words, and label mapping
MESSAGE_DIRECTORY = '20_newsgroups'
training_data = []
test_data = []
test_data_filenames = []
training_data_labels = []
test_data_labels = []
label_dictionary = {}

# NLTK Library stop words. Note: these words were taken directly from the NLTK library (www.geeksforgeeks.com)
stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", 
			  "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", 
			  "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", 
			  "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", 
			  "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", 
			  "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", 
			  "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", 
			  "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", 
			  "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", 
			  "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", 
			  "doing", "it", "how", "further", "was", "here", "than"]

# Mapping for label_dictionary = {<number>, <category>}
# 0 => alt.atheism
# 1 => comp.graphics
# 2 => comp.os.ms-windows.misc
# 3 => comp.sys.ibm.pc.hardware
# 4 => comp.sys.mac.hardware
# 5 => comp.windows.x
# 6 => misc.forsale
# 7 => rec.autos
# 8 => rec.motorcycles
# 9 => rec.sport.baseball
# 10 => rec.sport.hockey
# 11 => sci.crypt
# 12 => sci.electronics
# 13 => sci.med
# 14 => sci.space
# 15 => soc.religion.christian
# 16 => talk.politics.guns
# 17 => talk.politics.mideast
# 18 => talk.politics.misc
# 19 => talk.religion.misc


###########################################################################################################
# Class to create a Classifier object
###########################################################################################################
class Classifier(object):

		
	###########################################################################################################
	# Class Function to change message text to all lowercase and return an array of all words present. Uses
	# regular expression to parse the words. The translate function is used to remove punctuation.
	###########################################################################################################
	def parse(self, text):
		text = self.translate(text).lower()
		cleaned_words = re.split("\W+", text)
		return cleaned_words


	###########################################################################################################
	# Class Function to translate any punctuation in the message to blank space (remove it).
	###########################################################################################################
	def translate(self, text):
		words_wo_punctuation = str.maketrans("", "", string.punctuation)
		return text.translate(words_wo_punctuation)

		
	###########################################################################################################
	# Class Function to track the word counts in the message. Filters stop words as well.
	###########################################################################################################
	def get_word_totals(self, word_list):
		word_totals = {}

		for item in word_list:
			RE_D = re.compile('\d')
			if (item not in stop_words) and (not RE_D.search(item)) and len(item) < 50:
				word_totals[item] = word_totals.get(item, 0.0) + 1

		return word_totals

	
	###########################################################################################################
	# Class Function to initiate training of the classifier given the input data.
	###########################################################################################################
	def train(self, data, labels):
		self.message_totals = {}
		self.log_probability_message = {}
		self.word_totals = {}
		self.global_vocab = set()
		# Calculate total number of messages used for training
		n = len(training_data)

		# Iterate through each message category to calculate:
		# 1. Number of messages per category
		# 2. Log of the probability of each message category
		# 3. Create a nested word_count dictionary (3 dimensions)
		for category in label_dictionary:
			self.message_totals[label_dictionary.get(category)] = sum(1 for type in training_data_labels if type == category)
			self.log_probability_message[label_dictionary.get(category)] = math.log(self.message_totals[label_dictionary.get(category)] / n)
			self.word_totals[label_dictionary.get(category)] = {}

		# Iterate through each message in the set to populate word counts for the given category and also the global vocabulary based
		# on the contents and category of the current message.
		for data_item, label_item in zip(data, labels):
			category = label_dictionary[label_item]
			word_dictionary = self.get_word_totals(self.parse(data_item))
			for word, count in word_dictionary.items():

				# Add the word to the word totals dictionary and the global vocabulary if it doesn't already exist there
				if word not in self.word_totals[category]:
					self.word_totals[category][word] = 0.0

				if word not in self.global_vocab:
					self.global_vocab.add(word)
				
				# Populate the word_totals dictionary for the category of the message
				self.word_totals[category][word] += count

		# Print the global vocabulary to a .csv file
		if os.path.exists("Post_Processed_Data/GlobalVocabulary.csv"):
			os.remove("Post_Processed_Data/GlobalVocabulary.csv")
		header_info = ["_Global Vocabulary Words"]
		csvFile = open('Post_Processed_Data/GlobalVocabulary.csv', 'w', newline='')
		with csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(header_info)
			for word in self.global_vocab:
				writer.writerow([word])
		
		# Print the word counts for each category to multiple .csv files
		for category in label_dictionary:
			pathName = "Post_Processed_Data/Category_Word_Counts/" + label_dictionary.get(category) + "_WordCount.csv"
			if os.path.exists(pathName):
				os.remove(pathName)
			header_info = ["Word", "Word Count"]
			csvFile = open(pathName, 'w', newline='')
			with csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(header_info)
				for word, count in self.word_totals[label_dictionary.get(category)].items():
					writer.writerow([word, count])

		
	###########################################################################################################
	# Class Function to make message category predictions given the input data. Returns arrays of both the 
	# predicted categories and the actual categories for comparison purposes.
	###########################################################################################################
	def guess(self, data, data_filenames, labels):
		actual_results = []
		# Assign the category labels to the actual_results for accuracy comparison later
		for category in labels:
			actual_results.append(label_dictionary.get(category))

		prediction_results = []
		scores = {}
		denominators = {}

		vocab_length = len(self.global_vocab)
		message_count = 1

		# Print column header info in the results .csv file
		if os.path.exists("Post_Processed_Data/PredictionResults.csv"):
			os.remove("Post_Processed_Data/PredictionResults.csv")
		header_info = ["File Name", "Predicted Category", "Actual Category", "Result"]
		for category in label_dictionary:
			header_info.append("{} Probability".format(label_dictionary.get(category)))
		csvFile = open('Post_Processed_Data/PredictionResults.csv', 'w', newline='')
		with csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(header_info)

		for message, filename in zip(data, data_filenames):

			if message_count < len(data):
				print("Currently predicting message {}...".format(message_count), end = "\r")
			else:
				print("Currently predicting message {}...".format(message_count))

			# Get word counts for the message content and reset scores for each category to 0
			word_dictionary = self.get_word_totals(self.parse(message))

			for category in label_dictionary:
				scores[category] = 0
				denominators[category] = sum(self.word_totals[label_dictionary.get(category)].values()) + vocab_length

			for word, _ in word_dictionary.items():
				# Ignore word if it is not in the global vocabulary array (it was not present in any training data so it is meaningless)
				if word not in self.global_vocab: continue

				# Sum Log(p(word | category)) for the word in each category. Uses Laplacian Smoothing technique to eliminate the occurence of Log(0). Add to the overall score for the given category
				for category in label_dictionary:
					numerator = self.word_totals[label_dictionary.get(category)].get(word, 0.0) + 1
					log_word_given_category = math.log( numerator / denominators[category] )
					scores[category] += log_word_given_category
			
			# After all conditional probabilities calculated for each word and category, add the Log(p(category)) to each category score
			for category in label_dictionary:
				scores[category] += self.log_probability_message[label_dictionary.get(category)]
			
			# Find the highest score out of all categories and add it the prediction_results array. Then move onto the next message in the set
			highest_probablility = max(scores, key=scores.get)
			prediction_results.append(label_dictionary.get(highest_probablility))

			# Print the results data to an output .csv file
			if label_dictionary.get(highest_probablility) == actual_results[message_count-1]:
				result = "correct"
			else:
				result = "incorrect"
			row_info = [filename, label_dictionary.get(highest_probablility), actual_results[message_count-1], result]
			csvFile = open('Post_Processed_Data/PredictionResults.csv', 'a', newline='')
			with csvFile:
				writer = csv.writer(csvFile)
				for category in label_dictionary:
					row_info.append(scores[category])
				writer.writerow(row_info)

			message_count += 1

		return prediction_results, actual_results


###########################################################################################################
# Function to read all raw data from all files
###########################################################################################################
def read_messages():
	label_num = 0
	for subfolder in os.listdir(MESSAGE_DIRECTORY):

		# Populate a category dictionary with a numeric mapping to names of the subfolders
		label_dictionary[label_num] = subfolder

		file_split = len(next(os.walk(os.path.join(MESSAGE_DIRECTORY, subfolder)))[2]) // 2
		file_iterator = 0

		for file in os.listdir(os.path.join(MESSAGE_DIRECTORY, subfolder)):
			with open(os.path.join(MESSAGE_DIRECTORY, subfolder, file), encoding="latin-1") as f:
					# Read the entire contents of the file and assign it to the training or test data list along with the category label
					if file_iterator < file_split:
						training_data.append(f.read())
						training_data_labels.append(label_num)
					else:
						test_data_filenames.append(file)
						test_data.append(f.read())
						test_data_labels.append(label_num)

			file_iterator += 1
		label_num += 1


###########################################################################################################
# Main Entry Point for Script
###########################################################################################################
if __name__ == '__main__':

	# Read in all data from the set for both training and predictions
	print("Reading Messages...")
	read_messages()
	print("Finished Reading Messages!")

	# Instantiate a Classifier object
	bayes = Classifier()

	# Train the Classifier object with the first 500 files of each detected category
	print("Training Classifier...")
	bayes.train(training_data, training_data_labels)
	print("Finished Training Classifier!")

	# Predict the classification of the last 500 files of each detected category
	print("Predicting Message Categories...")
	predictions, actuals = bayes.guess(test_data, test_data_filenames, test_data_labels)
	print("Finished Predicting Message Categories!")

	# Calculate accuracy to 2 decimal places. If the prediction matches the actual category, increment the 
	# total correct guess count. Accuracy is defined as total correct guesses divided by the total number of 
	# messages predicted.
	correct_predictions = 0
	for predicted_category, actual_category in zip(predictions, actuals):
		if predicted_category == actual_category:
			correct_predictions += 1

	accuracy = correct_predictions / len(predictions) * 100
	print("Correct Predictions: {} , Incorrect Predictions: {} ".format(correct_predictions, len(predictions)-correct_predictions))
	print("Accuracy of classifier is: {0:.2f}%".format(accuracy))

	# Write output information to ResultsSummary.txt
	if os.path.exists("Post_Processed_Data/PredictionResultsSummary.txt"):
		os.remove("Post_Processed_Data/PredictionResultsSummary.txt")
	f = open("Post_Processed_Data/PredictionResultsSummary.txt","w+")
	f.write("Output Summary for run at {} \n\n".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
	f.write("Number of Training Messages: {} \n".format(len(training_data)))
	f.write("Number of Testing Messages: {} \n".format(len(test_data)))
	f.write("Correct Predictions: {} \n".format(correct_predictions))
	f.write("Incorrect Predictions: {} \n".format(len(predictions)-correct_predictions))
	f.write("Accuracy of classifier is: {0:.2f}%".format(accuracy))
	f.close()
	
	os.system("pause");