import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel(r'C:\Users\Acer\Downloads\spam_emails.xlsx')

# Function to clean the text
def cleaning(text):
    return re.sub(r'[^A-Za-z\s]', "", str(text))

# Clean the text and create lists for spam and non-spam
df['cleaned_text'] = df['text'].apply(cleaning)
list_spam = df[df['label'] == 'spam']['cleaned_text'].tolist()
list_non_spam = df[df['label'] != 'spam']['cleaned_text'].tolist()

# Split words
split_spam = [word for text in list_spam for word in text.split()]
split_non_spam = [word for text in list_non_spam for word in text.split()]

# Calculate prior probabilities
total_count = len(split_spam) + len(split_non_spam)
prior_spam = len(split_spam) / total_count
prior_non_spam = len(split_non_spam) / total_count

print(f'Prior Probability of Spam: {prior_spam:.2f}')
print(f'Prior Probability of Non-Spam: {prior_non_spam:.2f}')

# Create frequency distributions
from collections import Counter

spam_word_count = Counter(split_spam)
non_spam_word_count = Counter(split_non_spam)

# Function to calculate likelihoods
def calculate_likelihood(word, spam_count, non_spam_count):
    total_spam_words = len(split_spam)
    total_non_spam_words = len(split_non_spam)
    
    # Apply Laplace smoothing
    likelihood_spam = (spam_word_count[word] + 1) / (total_spam_words + len(spam_word_count))
    likelihood_non_spam = (non_spam_word_count[word] + 1) / (total_non_spam_words + len(non_spam_word_count))
    
    return likelihood_spam, likelihood_non_spam

# Function to classify a message
def classify(message):
    message_cleaned = cleaning(message)
    words = message_cleaned.split()
    
    prob_spam = prior_spam
    prob_non_spam = prior_non_spam
    
    for word in words:
        likelihood_spam, likelihood_non_spam = calculate_likelihood(word, spam_word_count, non_spam_word_count)
        prob_spam *= likelihood_spam
        prob_non_spam *= likelihood_non_spam
    
    return 'spam' if prob_spam > prob_non_spam else 'not spam'

# Example usage:
# Splitting dataset into train and test
train_data, test_data = train_test_split(df['cleaned_text'], test_size=0.2, random_state=42)
test_labels = df['label'][test_data.index]

# Evaluate the classifier
correct_predictions = 0

for message, true_label in zip(test_data, test_labels):
    prediction = classify(message)
    if prediction == true_label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print(f'Accuracy: {accuracy:.2f}')

# Example of classifying a new email
new_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."
classification_result = classify(new_email)

print(f'The email is classified as: {classification_result}')
