# Loading all necessary libraries

import matplotlib.pyplot as plt
import pandas as pd
import re
import csv
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
set(stopwords.words('english'))

# We are getting our data for categories and polarity, for the further sentiment analysis
category = pd.read_csv(r"C:/Users/Diana/Desktop/AUA/hpc/category.csv", header=None)
polarity = pd.read_csv(r"C:/Users/Diana/Desktop/AUA/hpc/polarity.csv", header=None)
df = pd.concat([category, polarity], axis=1)
df.columns = ['Category', 'Polarity']

# Here we are plotting the percentage of tweets for each category

fig1 = plt.figure()
df.Category.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('The percentage of tweets for each category')
fig1.savefig(r"C:/Users/Diana/Desktop/AUA/hpc/img1.png")
plt.show()
plt.close(fig1)

# This snippet of code generates the plot for the distribution of sentiments across all the tweets

fig2 = plt.figure()
df.Polarity.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
plt.title('The distribution of sentiments across all the tweets')
plt.ylabel(' ')
fig2.savefig(r"C:/Users/Diana/Desktop/AUA/hpc/img2.png")
plt.show()



# Finally, we draw the distribution of sentiment for each category
df.groupby(['Category', 'Polarity']).Polarity.count().unstack().plot(kind='bar')
plt.title('The distribution of sentiment for each category')
plt.grid(False)
plt.savefig(r"C:/Users/Diana/Desktop/AUA/hpc/img3.png")
plt.show()



# Cleaning our data from tweets

def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hash tags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub(r"\s+[a-zA-Z]\s+", " ", tweet)
    tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', tweet)  # remove punctuations
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    tweet = re.sub('(?<=[a-z])\'(?=[a-z])', '', tweet)
    tweet = tweet.replace('’', '')  # remove apostrophes
    return tweet


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"

                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)

# We get our data from the csv file then
# We are removing all emoticons and other redundant characters from our data
array = []
with open(r"C:/Users/Diana/Desktop/AUA/hpc/tweets.csv", encoding='utf-8-sig') as csv_file:
    df = csv.reader(csv_file)
    for row in df:
        row[0] = clean_tweet(row[0])
        row[0] = remove_emoji(row[0])
        row[0] = row[0].lower()
        array.append(row[0])

# Removing stopwords from the data set
data_set = ''.join(array)
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(data_set)
filtered_data = [w for w in word_tokens if not w in stop_words]
filtered_data = []

for w in word_tokens:
    if w not in stop_words:
        filtered_data.append(w)

final_data = ' '.join(filtered_data)

# Now we are getting 10 most frequent words from our cleaned data
# split() returns list of all the words in the string
split_it = final_data.split()

# We pass the split_it list to instance of Counter class, this counts the occurrence of each word in the string.
Counter = Counter(split_it)

# most_common(10) produces 10 frequently encountered words and their respective counts.
most_occur = Counter.most_common(10)

# We specify labels and frequencies for plotting the graph of 10 frequently encountered words.
label = []
frequency = []
for word in most_occur:
    label.append(word[0].capitalize())
    frequency.append(word[1])

# Here we are plotting the graph
labels = label[0:10]
fig4 = plt.figure(figsize=(10, 5))
plt.bar(labels, frequency[0:10], width=0.8, alpha=0.5, capsize=8, linewidth=1)
plt.xlabel("10 more frequent words", size=14)
plt.ylabel("Frequency", size=14)
plt.title('Word Frequencies', size=18)
plt.grid(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
fig4.savefig(r"C:/Users/Diana/Desktop/AUA/hpc/img4.png")
plt.show()
plt.close(fig4)

# Now we create and generate a word cloud image.
plt.figure(figsize=(10, 10))
wordcloud = WordCloud(width=800, height=500, max_font_size=80, max_words=150, background_color="white",
                      contour_color="steelblue", colormap="nipy_spectral").generate(final_data)
plt.imshow(wordcloud, interpolation="hermite")
plt.axis("off")
plt.show()

# Now we save the image
wordcloud.to_file(r"C:/Users/Diana/Desktop/AUA/hpc/wordcloud_img.png")
