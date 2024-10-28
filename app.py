import gradio as gr
import numpy as np
import pandas as pd
from Smart_Search2 import search_courses,generate_response

df = pd.read_csv('cleaned_analytics_vidhya_courses.csv')

embeddings = np.load('embeddings.npy')

def search_and_generate_response(query):
    print(f"Query received: {query}")
    results = search_courses(query, df, embeddings)
    print(f"Search results: {results}")

    output = []

    if results.empty:
        return "No results found."

    for idx, row in results.iterrows():
        output.append(f"**Title**: {row['Title']}\n**Description**: {row['combined_text']}\n**Relevance Score**: {row['similarity']:.2f}\n---")

    response = generate_response(query)
    print(f"Generated response: {response}")
    output.append(f"**Generated Response**: {response}")

    return "\n".join(output)

iface = gr.Interface(
    fn=search_and_generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your query here..."),
    outputs=gr.Textbox(),
    title="Analytics Vidhya Courses Smart Search",
    description="Search for Analytics Vidhya's free courses and get personalized responses based on your queries. Type your question in the box below and hit enter to see the results.",  # Description
)

iface.launch(share=True)
