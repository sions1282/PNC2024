"""
KOTE Sentiment Anaylsis

1. Pay attention to file names and paths.
2. Ensure that the virtual environment is set up and packages are installed (pip install datasets transformers pandas matplotlib numpy).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the model and tokenizer
model_name = "searle-j/kote_for_easygoing_people"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the text classification pipeline
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0,  # GPU number, set to -1 if using CPU
    top_k=None,  # Using top_k=None instead of return_all_scores
    function_to_apply='sigmoid'
)

# Load the CSV file
data_url = 'your_path/.csv' # Change to the desired media raw data (your_path~.csv)
df = pd.read_csv(data_url)

# Define emotion labels (ensure this matches your model output)
emotion_labels = ['불평/불만', '환영/호의', '감동/감탄', '지긋지긋', 
                  '고마움', '슬픔', '화남/분노', '존경', '기대감', 
                  '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', 
                  '뿌듯함', '편안/쾌적', '신기함/관심', '아껴주는', 
                  '부끄러움', '공포/무서움', '절망', '한심함', 
                  '역겨움/징그러움', '짜증', '어이없음', '없음', 
                  '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', 
                  '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', 
                  '당황/난처', '경악', '부담/안_내킴', '서러움', 
                  '재미없음', '불쌍함/연민', '놀람', '행복', 
                  '불안/걱정', '기쁨', '안심/신뢰']

# Create columns to store analysis results
for i in range(1, 6): 
    df[f'{i}위'] = ""

# List to store rows exceeding token limit
exceeded_token_rows = []

# Perform emotion analysis
for index, row in df.iterrows():
    sentence = row["내용"]
    if isinstance(sentence, str):
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > 512:
            exceeded_token_rows.append(index)
            df.at[index, '1위'] = 'Token count exceeded'
        else:
            try:
                scores = pipe(sentence)[0]
                filtered_scores = [item for item in scores if item['score'] > 0.5]
                filtered_scores.sort(key=lambda x: x['score'], reverse=True)
                for i, item in enumerate(filtered_scores):
                    if i < 5:
                        df.at[index, f'{i+1}위'] = f"{item['label']}: {item['score']:.2f}"
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                exceeded_token_rows.append(index)
                df.at[index, '1위'] = 'Processing error'
    else:
        df.at[index, '1위'] = 'Invalid content'

# Save the updated DataFrame to a CSV file
output_path = 'your_path/.csv' # Change to the desired media raw data (your_path~.csv)
df.to_csv(output_path, index=False)

# Load the DataFrame for visualization
df = pd.read_csv(output_path)

# Function to split emotion label and percentage
def split_label(label):
    if pd.isna(label):
        return '', 0.0
    if ':' in label:
        emotion, percent = label.split(':')
        return emotion.strip(), float(percent.strip())
    return '', 0.0

# Define a dictionary for emotion translations (keep original labels)
emotion_translation = {
    '불평/불만': 'Complaint/Dissatisfaction',
    '환영/호의': 'Welcome/Friendliness',
    '감동/감탄': 'Impression/Admiration',
    '지긋지긋': 'Tedious',
    '고마움': 'Gratitude',
    '슬픔': 'Sadness',
    '화남/분노': 'Anger',
    '존경': 'Respect',
    '기대감': 'Expectation',
    '우쭐댐/무시함': 'Arrogance/Disdain',
    '안타까움/실망': 'Regret/Disappointment',
    '비장함': 'Heroic',
    '의심/불신': 'Doubt/Distrust',
    '뿌듯함': 'Pride',
    '편안/쾌적': 'Comfortable/Pleasant',
    '신기함/관심': 'Curiosity/Interest',
    '아껴주는': 'Caring',
    '부끄러움': 'Shame/Embarrassment',
    '공포/무서움': 'Fear',
    '절망': 'Despair',
    '한심함': 'Pitiful',
    '역겨움/징그러움': 'Disgust',
    '짜증': 'Annoyance',
    '어이없음': 'Absurdity',
    '없음': 'None',
    '패배/자기혐오': 'Defeat/Self-Loathing',
    '귀찮음': 'Annoyance',
    '힘듦/지침': 'Tiredness/Exhaustion',
    '즐거움/신남': 'Joy/Excitement',
    '깨달음': 'Enlightenment',
    '죄책감': 'Guilt',
    '증오/혐오': 'Hatred',
    '흐뭇함(귀여움/예쁨)': 'Delight (Cuteness/Beauty)',
    '당황/난처': 'Embarrassment',
    '경악': 'Shock',
    '부담/안_내킴': 'Burden/Reluctance',
    '서러움': 'Sorrow',
    '재미없음': 'Boredom',
    '불쌍함/연민': 'Pity/Compassion',
    '놀람': 'Surprise',
    '행복': 'Happiness',
    '불안/걱정': 'Anxiety/Worry',
    '기쁨': 'Joy',
    '안심/신뢰': 'Relief/Trust'
}

# Assign colors for visualization
positive_emotions = [
    'Welcome/Friendliness', 'Impression/Admiration', 'Gratitude', 'Respect', 'Expectation', 'Pride',
    'Comfortable/Pleasant', 'Curiosity/Interest', 'Caring', 'Joy/Excitement', 'Enlightenment',
    'Delight (Cuteness/Beauty)', 'Surprise', 'Happiness', 'Joy', 'Relief/Trust'
]

negative_emotions = [
    'Complaint/Dissatisfaction', 'Tedious', 'Sadness', 'Anger', 'Arrogance/Disdain', 'Regret/Disappointment',
    'Heroic', 'Doubt/Distrust', 'Shame/Embarrassment', 'Fear', 'Despair', 'Pitiful',
    'Disgust', 'Annoyance', 'Absurdity', 'Defeat/Self-Loathing', 'Annoyance', 'Tiredness/Exhaustion',
    'Guilt', 'Hatred', 'Embarrassment', 'Shock', 'Burden/Reluctance', 'Sorrow', 'Boredom',
    'Pity/Compassion', 'Anxiety/Worry'
]

colors_bar = {emotion: '#f1e7c6' for emotion in positive_emotions}
colors_bar.update({emotion: '#760023' for emotion in negative_emotions})
colors_bar['None'] = 'gray'

# Extract all emotion labels and percentages
emotions = []
for col in ['1위', '2위', '3위', '4위', '5위']:
    emotions += df[col].apply(split_label).tolist()

# Convert to DataFrame
emotions_df = pd.DataFrame(emotions, columns=['Emotion', 'Percent'])
emotions_df = emotions_df[emotions_df['Emotion'] != '']

# Translate emotion labels to English
emotions_df['Emotion'] = emotions_df['Emotion'].map(emotion_translation)

# Calculate average percentage per emotion
mean_emotions = emotions_df.groupby('Emotion')['Percent'].mean().sort_values(ascending=False)

# Select top 10 emotions
top_10_emotions = mean_emotions.head(10)[::-1]  # Reverse the order

# Map colors to emotions
top_10_colors = [colors_bar[emotion] for emotion in top_10_emotions.index]

# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
plt.barh(top_10_emotions.index, top_10_emotions.values, color=top_10_colors)
plt.xlabel('Average Percentage')
plt.ylabel('Emotion')
plt.title('Average Percentage of Top 10 Emotions from Top 5 Labels')
plt.xlim(0.5, top_10_emotions.max())
plt.xticks(np.arange(0.5, top_10_emotions.max() + 0.1, 0.1))
plt.tight_layout(pad=2.0)  # Automatically adjust the layout
output_dir = os.path.dirname(output_path)
plt.savefig(os.path.join(output_dir, 'Webtoon_top_10_emotions_bar.png'), bbox_inches='tight', transparent=True, dpi=300)  # Save bar chart as image
plt.show()

# Extract data from the '1위' column and create a pie chart
first_rank_emotions = df['1위'].apply(split_label).tolist()
first_rank_emotions_df = pd.DataFrame(first_rank_emotions, columns=['Emotion', 'Percent'])
first_rank_emotions_df = first_rank_emotions_df[first_rank_emotions_df['Emotion'] != '']

# Translate emotion labels to English
first_rank_emotions_df['Emotion'] = first_rank_emotions_df['Emotion'].map(emotion_translation)

# Calculate the sum of percentages per emotion
sum_first_rank_emotions = first_rank_emotions_df.groupby('Emotion')['Percent'].sum()

# Select the top 5 emotions
top_5_emotions = sum_first_rank_emotions.nlargest(5)

# Define colors for pie chart
colors = {
    'Impression/Admiration': '#f1e7c6',
    'Expectation': '#dfd09d',
    'Enlightenment': '#d7c17a',
    'Joy/Excitement': '#c7ac54',
    'Delight (Cuteness/Beauty)': '#b89b39',
    'Regret/Disappointment': '#760023',
    'Absurdity': '#9e002f',
    'None': 'gray',
    'Anger': '#c10039'
}

# Map colors to top 5 emotions
top_5_colors = [colors.get(emotion, 'black') for emotion in top_5_emotions.index]

# Create a pie chart
plt.figure(figsize=(10, 6))
top_5_emotions.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=top_5_colors)
plt.ylabel('')
plt.title('Distribution of Top 5 Emotions in the First Rank')
plt.tight_layout(pad=2.0)  # Automatically adjust the layout
plt.savefig(os.path.join(output_dir, 'Webtoon_top_5_emotions_pie.png'), bbox_inches='tight', transparent=True, dpi=300)  # Save pie chart as image
plt.show()