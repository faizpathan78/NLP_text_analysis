import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Step 1: Data Extraction

# Load the 'Input.xlsx' file to get the list of URLs and URL_IDs
input_file = 'Input.xlsx'
output_folder = 'output_text_files/'  # Folder to save extracted text files

# Read the input Excel file
input_df = pd.read_excel("Input.xlsx")

# Create a folder to store the extracted text files if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to extract article text from a URL
def extract_article_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Implement logic to extract only the article content
            article_content = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            return article_content
        else:
            print(f"Failed to retrieve content from URL: {url}")
            return None
    except Exception as e:
        print(f"An error occurred while processing URL: {url}")
        print(e)
        return None

# Iterate through each row in the input Excel file
for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract the article text from the URL
    article_text = extract_article_text(url)

    if article_text:
        # Save the extracted text into individual text files
        output_file_path = os.path.join(output_folder, f'{url_id}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(article_text)
        print(f"Extracted and saved text from URL {url_id}")

print("Data extraction and text file creation completed.")

# Step 2: Load stopwords, positive words, and negative words

# Load stopwords from the 'Stopwords' folder
stop_words_folder = 'Stopwords/'
stop_words_files = [
    'StopWords_Names.txt',
    'StopWords_Geographic.txt',
    'StopWords_GenericLong.txt',
    'StopWords_Generic.txt',
    'StopWords_DatesandNumbers.txt',
    'StopWords_Currencies.txt',
    'StopWords_Auditor.txt'
]

stop_words = set()
for file in stop_words_files:
    with open(os.path.join(stop_words_folder, file), 'r', encoding='utf-8') as f:
        stop_words.update(f.read().splitlines())

# Load positive words from the 'MasterDictionary' folder
positive_words_file = 'positive-words.txt'
with open(os.path.join('MasterDictionary', positive_words_file), 'r', encoding='utf-8') as f:
    positive_words = set(f.read().splitlines())

# Load negative words from the 'MasterDictionary' folder
negative_words_file = 'negative-words.txt'
with open(os.path.join('MasterDictionary', negative_words_file), 'r', encoding='utf-8') as f:
    negative_words = set(f.read().splitlines())

# Step 3: Performing Sentiment Analysis
def perform_sentiment_analysis(article_text, stop_words, positive_words, negative_words):
    # Replace the following lines with your actual sentiment analysis logic
    cleaned_text = clean_text(article_text, stop_words)
    positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(
        cleaned_text, positive_words, negative_words
    )
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to clean text using stop words
def clean_text(text, stop_words):
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(cleaned_words)

# Function to calculate sentiment scores
def calculate_sentiment_scores(text, positive_words, negative_words):
    words = word_tokenize(text)
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to calculate average sentence length
def average_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if len(sentences) > 0 else 0

# Function to count complex words
def count_complex_words(text):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if len(word) > 2)  # Adjust the condition for complexity
    return complex_word_count

# Function to count syllables in a word
def count_syllables(word):
    # Convert word to lowercase for consistent counting
    word = word.lower()

    # Count the number of vowels in the word
    num_vowels = 0
    for char in word:
        if char in "aeiouy":
            num_vowels += 1

    # Handle special cases of endings
    if word.endswith("es"):
        num_vowels -= 1
    if word.endswith("ed"):
        num_vowels -= 1

    # Ensure that the minimum count is at least 1
    if num_vowels == 0:
        num_vowels = 1

    return num_vowels

# Function to calculate average word length
def average_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    return total_characters / len(words)

# Function to count personal pronouns
def count_personal_pronouns(text):
    personal_pronoun_count = len(re.findall(r'\b(I|we|my|ours|us)\b', text, flags=re.IGNORECASE))
    return personal_pronoun_count

# Step 4: Iterate through each article, perform sentiment analysis, and save results

# Create a DataFrame for the output structure
output_df = pd.DataFrame(columns=[
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
    'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
])

for index, row in input_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Read the content of the text file (assumes you've saved them in Step 1)
    try:
        with open(f'output_text_files/{url_id}.txt', 'r', encoding='utf-8') as file:
            article_text = file.read()
    except FileNotFoundError:
        print(f"File not found for URL {url_id}")
        continue

    # Perform sentiment analysis and text analysis
    positive_score, negative_score, polarity_score, subjectivity_score = perform_sentiment_analysis(
        article_text, stop_words, positive_words, negative_words
    )
    avg_sentence_len = average_sentence_length(article_text)
    complex_word_count = count_complex_words(article_text)
    word_count = len(word_tokenize(clean_text(article_text, stop_words)))
    personal_pronoun_count = count_personal_pronouns(article_text)
    avg_word_len = average_word_length(clean_text(article_text, stop_words))

    # Calculate the percentage of complex words
    percentage_complex_words = (complex_word_count / word_count) * 100 if word_count > 0 else 0

    # Calculate the average number of syllables per word
    syllables_per_word = sum(count_syllables(word) for word in word_tokenize(article_text)) / word_count

    # Calculate the FOG Index
    fog_index = 0.4 * (avg_sentence_len + percentage_complex_words)

    # Calculate the average number of words per sentence
    avg_words_per_sentence = word_count / len(sent_tokenize(article_text)) if len(sent_tokenize(article_text)) > 0 else 0

    # Save the computed variables into the output DataFrame
    output_df = pd.concat([output_df, pd.DataFrame({
        'URL_ID': [url_id],
        'URL': [url],
        'POSITIVE SCORE': [positive_score],
        'NEGATIVE SCORE': [negative_score],
        'POLARITY SCORE': [polarity_score],
        'SUBJECTIVITY SCORE': [subjectivity_score],
        'AVG SENTENCE LENGTH': [avg_sentence_len],
        'PERCENTAGE OF COMPLEX WORDS': [percentage_complex_words],
        'FOG INDEX': [fog_index],
        'AVG NUMBER OF WORDS PER SENTENCE': [avg_words_per_sentence],
        'COMPLEX WORD COUNT': [complex_word_count],
        'WORD COUNT': [word_count],
        'SYLLABLE PER WORD': [syllables_per_word],
        'PERSONAL PRONOUNS': [personal_pronoun_count],
        'AVG WORD LENGTH': [avg_word_len]
    })], ignore_index=True)

# Save the output DataFrame to the Excel file
output_df.to_excel('Output Data Structure.xlsx', index=False, header=True)
