import pickle
import re
import string
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
app = Flask(__name__)

# Dummy data for news articles
news_articles = [
    {"title": "News Article 1", "content": "Content for article 1..."},
    {"title": "News Article 2", "content": "Content for article 2..."},
    # Add more articles as needed
]

df_news2 = pd.read_csv('news_dataset/MINDsmall_train/news.tsv', sep='\t', header=None)
df_news2.columns = ['News ID',
                    'Category',
                    'SubCategory',
                    'Title',
                    'Abstract',
                    'URL',
                    'Title Entities',
                    'Abstract Entities']

df_news = df_news2.head(10000)
# Handle missing values in the "Title" and "Abstract" columns
df_news.loc[:, 'Title'] = df_news['Title'].fillna('')
df_news.loc[:, 'Abstract'] = df_news['Abstract'].fillna('')
df_news.loc[:, 'Text'] = df_news['Title'].fillna('') + ' ' + df_news['Abstract'].fillna('')
pd.set_option('mode.chained_assignment', None)


# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined text column
tfidf_matrix = tfidf_vectorizer.fit_transform(df_news['Text'])

# Calculate cosine similarity between articles
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get top N recommendations for a given article
def get_recommendations(article_id, top_n=10):
    category = df_news.at[article_id, 'Category']
    sim_scores = list(enumerate(cosine_sim[article_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_articles = [i[0] for i in sim_scores if df_news.at[i[0], 'Category'] == category][:top_n]
    return df_news.iloc[recommended_articles]

# Loading the model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    loaded_LR = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorization = pickle.load(vectorizer_file)

def word_proc(text):
    '''it takes a text as input and performs various text preprocessing steps, such as converting the text to lowercase, removing square brackets and their content, 
    replacing non-word characters with spaces, removing URLs, removing HTML tags, removing punctuation, removing newline characters, and removing words with digits.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def detect_fake_news(text_input):
    # Use the loaded model and vectorizer for prediction
    testing_news = {"text":[text_input]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_proc) 
    new_x_test = new_def_test["text"]
    new_xv_test = loaded_vectorization.transform(new_x_test)
    pred_LR = loaded_LR.predict(new_xv_test)

    return ("Prediction: {} ".format(output_lable(pred_LR[0]) ))


@app.route('/')
def index():
    # Get recommended articles
    recommended_articles = get_recommendations(article_id=0, top_n=1)
    return render_template('index.html', articles=recommended_articles)

@app.route('/detect_fake_news', methods=['POST'])
def detect_fake_news_route():
    recommended_articles = get_recommendations(article_id=0, top_n=1)
    text_input = request.form['news_input']
    detection_result = detect_fake_news(text_input)
    return render_template('index.html', articles=recommended_articles, detection_result=detection_result)



@app.route('/search', methods=['POST'])
def search_news_route():
    search_query = request.form['search_input'].lower()
    search_result_article_id = df_news[df_news.apply(lambda row: search_query in (row['Title'].lower(),row['Category'].lower(), row['SubCategory'].lower(), row['Abstract'].lower()), axis=1)].index[0]
    recommended_articles = get_recommendations(article_id=search_result_article_id, top_n=10) 
    return render_template('index.html', search_results=recommended_articles.to_dict(orient='records'))



if __name__ == '__main__':
    app.run(debug=True)
