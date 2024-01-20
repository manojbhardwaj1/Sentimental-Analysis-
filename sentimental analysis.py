import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

# Read in data
df = pd.read_csv("D:\\DDownloads\\Musical_instruments_reviews.csv")
print(df.shape)
df.head()

# Check for missing values
print(df.isnull().sum())

# Check for NaN values
print(df.isna().sum())

target_column = 'overall'
count_result = df[target_column].value_counts()
print(count_result)

#Exploratory Data Analysis
ax = df['overall'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by rating ',
          figsize=(10, 5))
ax.set_xlabel('Review of ratings')
plt.show()

example = df["reviewText"].values[13]
print(example)
print(example.split(" "))
len(example.split(" "))

tokens = nltk.word_tokenize(example)
print(tokens)
len(tokens)

tagged = nltk.pos_tag(tokens)
tagged[:22]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# VADER (Valence Aware Dictionary and sEntiment Reasoner) Seniment Scoring

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores("examples")
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores('I am so happy!')

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['reviewText']
    myid = row['reviewerID']
    
    # Check if 'reviewText' is a string before calling polarity_scores
    if isinstance(text, str):
        res[myid] = sia.polarity_scores(text)
    else:
        # Handle the case when 'reviewText' is not a string (skip or set a default value)
        res[myid] = None  # You can set a default value, or simply skip it using 'continue'

# Now 'res' contains polarity scores for text rows that were strings.


# Convert the 'res' dictionary to a DataFrame and transpose it (swap rows and columns)
vaders = pd.DataFrame(res).T
# Set the 'Id' column as the index and rename the 'index' column to 'Id'
vaders = vaders.reset_index().rename(columns={'index': 'Id'}).set_index('Id')
# Merge the 'vaders' DataFrame with the 'df' DataFrame on the 'Id' column (not 'ID')
merged_df = vaders.merge(df, how='left', left_index=True, right_on='reviewerID')

# Now we have sentiment score and metadata
vaders.head()

# Plot VADER results
ax = sns.barplot(data=vaders, x='vader_pos', y='vader_compound')
ax.set_title('Compound vs Positive')
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='vader_compound', y='vader_pos', ax=axs[0])
sns.barplot(data=vaders, x='vader_compound', y='vader_neu', ax=axs[1])
sns.barplot(data=vaders, x='vader_compound', y='vader_neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


#Roberta Pretrained Model

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# VADER results on example
print(example)
sia.polarity_scores(example)

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['reviewText']
        myid = row['reviewerID']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for reviewerID {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'reviewerID'})
results_df = results_df.merge(df, how='left')

# Compare Scores between models
results_df.columns

# Combine and compare
sns.pairplot(data=results_df, 
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='vader_compound',
            palette='tab10')
plt.show()

# Review Examples:
# Positive 1-Star and Negative 5-Star Reviews

results_df.query('overall == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('overall == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

# nevative sentiment 5-Star view
results_df.query('overall == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

results_df.query('overall == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0] 

