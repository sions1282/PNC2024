"""
Word Cloud

1. Pay attention to file names, fonts, and file paths.
2. Use the code after setting up the virtual environment and downloading the packages. (pip install pandas matplotlib wordcloud requests)
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import os

# Font settings
font_path = 'your_path/NanumGothic.ttf' # Change the font path (your_path)

if os.path.isfile(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:
    raise FileNotFoundError(f"Font file not found: {font_path}")

# Raw data settings
file_path = 'your_path/.csv' # Change to the desired media raw data (your_path~.csv)
data = pd.read_csv(file_path)

# Stopwords settings and application
stopwords_url = 'your_path/stopwords.csv'  # Replace with your stopwords file path (your_path)
stopwords_data = pd.read_csv(stopwords_url)
stopwords = stopwords_data['불용어'].tolist()
filtered_data = data[~data['단어'].isin(stopwords)]

# POS settings and application
target_pos = ['NNG', 'NNP', 'VA', 'VA-R', 'VA-I']
filtered_data = filtered_data[filtered_data['품사'].isin(target_pos)]

# Word and frequency extraction
word_freq = dict(zip(filtered_data['단어'], filtered_data['빈도수']))

# Select top 75 words
top_75_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:75])

# Color assignment
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    frequency = filtered_data[filtered_data['단어'] == word]['빈도수'].values[0]
    if filtered_data[filtered_data['단어'] == word]['품사'].values[0] in ['NNG', 'NNP']:
        # Nouns: Red hues
        if frequency < 100:
            return "#c10039"
        elif 100 <= frequency < 200:
            return "#9e002f"
        elif 200 <= frequency < 500:
            return "#760023"
        else:
            return "#4c0016"
    else:
        # Adjectives: Teal hues
        if frequency < 100:
            return "#009999" 
        elif 100 <= frequency < 200:
            return "#007777"
        elif 200 <= frequency < 500:
            return "#005555"
        else:
            return "#003333"

# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    font_path=font_path,  
    color_func=color_func,
    max_words=75 
).generate_from_frequencies(top_75_words)

# Visualize and save word cloud
plt.figure(figsize=(10, 8)) 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.text(0.5, 1.1, 'Word Cloud', fontsize=20, ha='center', va='center', transform=plt.gca().transAxes, fontproperties=fm.FontProperties(fname=font_path))

output_path = 'your_path/WordCloud.png'  # Change the output path (your_path)
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', transparent=True)

plt.show()