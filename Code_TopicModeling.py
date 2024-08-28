"""
LDA Topic Modeling

1. Pay attention to file names and paths.
2. Ensure that the virtual environment is set up and packages are installed (pip install pandas numpy kiwipiepy gensim matplotlib).
"""

import pandas as pd
import numpy as np
from kiwipiepy import Kiwi
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# Font settings
font_path = 'your_path/NanumGothic.ttf' # Change the font path (your_path)

if os.path.isfile(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:
    raise FileNotFoundError(f"Font file not found: {font_path}")

# Load data
data_url = "your_path/your_data.csv"  # Replace with your data file path (your_path)
df = pd.read_csv(data_url)

# Function to add user dictionary
def add_user_dictionary(kiwi, user_dict_url):
    user_dict_df = pd.read_csv(user_dict_url)
    for idx, row in user_dict_df.iterrows():
        word = row['Word']
        pos = row['POS']
        kiwi.add_user_word(word, pos)

# Initialize Kiwi morphological analyzer
kiwi = Kiwi()
user_dict_url = "your_path/user_dictionary.csv"  # Replace with your user dictionary file path (your_path)
add_user_dictionary(kiwi, user_dict_url)

# Load synonym and POS unification dictionaries
synonyms_url = "your_path/synonyms.csv"  # Replace with your synonyms file path (your_path)
unify_pos_url = "your_path/unify_pos.csv"  # Replace with your POS unification file path (your_path)
stopwords_url = "your_path/stopwords.csv"  # Replace with your stopwords file path (your_path)

# Load synonym dictionary
synonyms_df = pd.read_csv(synonyms_url)
synonym_dict = {}
for idx, row in synonyms_df.iterrows():
    group = row['Group']
    words = row['Words'].split(', ')  # Split by comma and space
    for word in words:
        synonym_dict[word] = group

# Load POS unification dictionary
unify_pos_df = pd.read_csv(unify_pos_url)
unify_pos_dict = {}
for idx, row in unify_pos_df.iterrows():
    word = row['Word']
    original_pos = row['Original_POS']
    unified_pos = row['Unified_POS']
    unify_pos_dict[(word, original_pos)] = unified_pos

# Load stopwords list
stopwords_df = pd.read_csv(stopwords_url)
stopwords = set(stopwords_df['Stopwords'])

# Add morphological analysis results to the dataframe
def morph_analysis(x):
    if isinstance(x, str):
        tokens = kiwi.tokenize(x)
        new_tokens = []
        for token in tokens:
            unified_pos = unify_pos_dict.get((token.form, token.tag), token.tag)
            new_tokens.append((token.form, unified_pos))
        return new_tokens
    return None

df['Morph_Analysis'] = df['Content'].apply(morph_analysis)

# Define preprocessing function (using only nouns)
def preprocess(tokens):
    if tokens is None:
        return []
    filtered_tokens = [token[0] for token in tokens if token[1] in ['NNG', 'NNP'] and token[0] not in stopwords]
    # Apply synonym dictionary
    filtered_tokens = [synonym_dict.get(token, token) for token in filtered_tokens]
    return filtered_tokens

# Extract and preprocess tokens from morphological analysis results
df['tokens'] = df['Morph_Analysis'].apply(preprocess)

# Create dictionary and corpus for Gensim
dictionary = corpora.Dictionary(df['tokens'].dropna())
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens'].dropna()]
all_tokens = df['tokens'].tolist()

# Determine the number of topics
coherence_values = []
for i in range(2, 7):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary, passes=10, iterations=100)
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=all_tokens, dictionary=dictionary, topn=10)
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_values.append(coherence_lda)

x = range(2, 7)
plt.plot(x, coherence_values)
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.show()

# Train the LDA model
num_topics = 4  # Number of topics to extract

# Variable to store results
results = []

# Run multiple times and save results
for _ in range(10):  # Repeat as many times as needed
    random_state = np.random.randint(0, 10000)
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=random_state)

    # Print topics and convert to DataFrame
    topic_list = []
    for idx, topic in lda_model.print_topics(-1):
        topic_dict = {'Topic': idx, 'Words': topic}
        topic_list.append(topic_dict)

    # Save results
    results.append((random_state, topic_list))

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Display all results so the user can choose the desired result
for random_state, topic_list in results:
    print(f"Random State: {random_state}")
    topics_df = pd.DataFrame(topic_list)
    print(topics_df)
    print("\n")

# Enter the desired random_state from the list above to print the corresponding result
desired_random_state = int(input("Enter the desired random_state from the list above: "))

for random_state, topic_list in results:
    if random_state == desired_random_state:
        topics_df = pd.DataFrame(topic_list)
        print(f"Random State: {random_state}")
        print(topics_df)
        break
else:
    print("No matching random_state found.")

# Save results as a CSV file
output_path = "your_path/output_topics.csv"  # Replace with your output file path (your_path)
topics_df.to_csv(output_path, index=False)