# -*- coding: utf-8 -*-


import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url="https://courses.analyticsvidhya.com"

course_items=[]

for i in range(1, 9):
    r = requests.get(f"https://courses.analyticsvidhya.com/collections/courses?page={i}")
    soup = BeautifulSoup(r.content, "html.parser")

    course_link = soup.find_all("li", class_="products__list-item")

    for item in course_link:
        for link in item.find_all("a", href=True):
            full_link = base_url + link["href"]
            course_items.append(full_link)

print(len(course_items))

print(course_items)

titles = []
descriptions = []
course_titles = []
lessons_list = []

for course_link in course_items:
    r = requests.get(course_link)
    soup = BeautifulSoup(r.content, "html.parser")

    title = soup.find('h1', class_='section__heading')
    if title:
        titles.append(title.get_text(strip=True))
    else:
        titles.append("Title not found")

    description_div = soup.find('div', class_='fr-view')
    if description_div and description_div.find('p'):
        description = description_div.find('p').get_text(strip=True)
        descriptions.append(description)
    else:
        descriptions.append("Description not found")

    course_title_tag = soup.find('h5', class_='course-curriculum__chapter-title')
    if course_title_tag:
        course_titles.append(course_title_tag.get_text(strip=True))
    else:
        course_titles.append("Course title not found")

    lessons = []
    lesson_tags = soup.find_all('span', class_='course-curriculum__chapter-lesson')
    if lesson_tags:
        for lesson in lesson_tags:
            lessons.append(lesson.get_text(strip=True))
        lessons_list.append(", ".join(lessons))
    else:
        lessons_list.append("Lessons not found")

data = {
    "Title": titles,
    "Description": descriptions,
    "Course Title": course_titles,
    "Lessons": lessons_list
}

df = pd.DataFrame(data)

df.to_csv("courses.csv", index=False)

df=pd.read_csv("courses.csv")
df.head()



from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text, use_lemmatization=False):
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    if use_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    else:
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def preprocess_data(df):
    df['cleaned_description'] = df['description'].apply(preprocess_text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['cleaned_description'].tolist(), show_progress_bar=True)
    np.save('embeddings.npy', embeddings)
    return embeddings, model





import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

client = chromadb.Client()
collection = client.create_collection("analytics_embeddings")

def preprocess_data(df, text_columns=['Description', 'Curriculum'], output_file='cleaned_data.csv'):
    missing_columns = [col for col in text_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")

    df['combined_text'] = df[text_columns].apply(lambda row: ' '.join([preprocess_text(str(text)) for text in row]), axis=1)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
    np.save("embeddings.npy", embeddings)

    client.delete_collection("analytics_embeddings")

    collection = client.create_collection("adityaa")

    ids = [str(i) for i in range(len(embeddings))]
    try:
        collection.add(
            documents=df['combined_text'].tolist(),
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=[{"title": df['Title'].iloc[i], "id": i} for i in range(len(df))]
        )
    except Exception as e:
        print(f"Error adding embeddings to ChromaDB: {e}")
        return None, None

    df.to_csv(output_file, index=False)

    print(f"Cleaned data saved to {output_file}")

    return embeddings, model

df = pd.read_csv('courses.csv')

embeddings, model = preprocess_data(df, text_columns=['Description', 'Lessons'], output_file='cleaned_analytics_vidhya_courses.csv')

df1=pd.read_csv("cleaned_analytics_vidhya_courses.csv")
df1.head()

print(df[['Description', 'Lessons', 'combined_text']].head())
print(f'Embeddings shape: {embeddings.shape}')



import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai

def search_courses(query, df, embeddings):
    query_embedding = model.encode([query])[0]

    similarities = cosine_similarity([query_embedding], embeddings)[0]

    df['similarity'] = similarities

    results = df.sort_values(by='similarity', ascending=False).head(5)

    return results



openai.api_key = 'YOUR OPENAI API KEY'

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']













