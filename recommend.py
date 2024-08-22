import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

# Combine the "Title" and "Abstract" columns into a single text column
df_news['Text'] = df_news['Title'].fillna('') + ' ' + df_news['Abstract'].fillna('')

# Define the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined text column
tfidf_matrix = tfidf_vectorizer.fit_transform(df_news['Text'])

# Calculate cosine similarity between articles
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get top N recommendations for a given article
def get_recommendations(article_id, top_n=10):
    # Get the category of the given article
    category = df_news.at[article_id, 'Category']
    
    # Get the pairwise cosine similarity scores for the article
    sim_scores = list(enumerate(cosine_sim[article_id]))
    
    # Sort the articles based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N similar articles from the same category
    recommended_articles = [i[0] for i in sim_scores if df_news.at[i[0], 'Category'] == category][:top_n]
    
    return df_news.iloc[recommended_articles]

# User provides a category or subcategory as input
user_input_category = "Lifestyle"  # Replace with user's input

# Filter the dataset based on the selected category
filtered_data = df_news[df_news.apply(lambda row: user_input_category.lower() in (row['Category'], row['SubCategory']), axis=1)]

# Check if there are articles in the selected category
if not filtered_data.empty:
    # Define the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the TF-IDF vectorizer on the combined text column
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['Text'])

    # Calculate cosine similarity between articles
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to get top N recommendations based on the selected category
    def get_category_recommendations(top_n=10):
        # Select one article from the filtered data (you can choose any)
        article_id = filtered_data.iloc[0].name

        # Get the pairwise cosine similarity scores for the selected article
        sim_scores = list(enumerate(cosine_sim[article_id]))

        # Sort the articles based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top N similar articles in the same category
        recommended_articles = [i[0] for i in sim_scores][:top_n]

        return filtered_data.iloc[recommended_articles]

    # Example usage: Get top 10 recommendations based on the selected category
    recommended_articles = get_category_recommendations(top_n=10)
    print(recommended_articles[['Title', 'Abstract']])
else:
    print("No articles found in the selected category.")
