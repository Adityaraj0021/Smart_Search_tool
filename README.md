
# Smart Search Tool for Analytics Vidhya Courses
## Introduction
The Smart Search Tool enables users to search and retrieve relevant information on free courses from the Analytics Vidhya platform using natural language queries. The tool generates contextual responses with a generative language model, making it user-friendly and efficient for exploring course content.

## Problem Definition
The main objective was to build a search tool that efficiently filters a dataset of Analytics Vidhya courses based on user queries. It responds with relevant, concise information to match user needs, leveraging a language model for generating context-aware responses.

## Data Collection
### Web Scraping with BeautifulSoup
Data for this tool was collected by web scraping the Analytics Vidhya website to gather information on free courses. BeautifulSoup was used in conjunction with requests to extract the following details:

Course titles

Descriptions

Curriculum details

The extracted data is saved to a CSV file for processing and embedding generation.

## Data Preprocessing
Text Cleaning: Consolidated and cleaned course descriptions and curriculum content.

Embedding Generation: Generated sentence embeddings using the SentenceTransformer model (all-MiniLM-L6-v2), optimized for semantic textual similarity and fast processing.
## Model Selection
### Embedding Model
The all-MiniLM-L6-v2 model was selected for embedding generation due to its balance of speed and accuracy, crucial for semantic search tasks.

### Language Model
GPT-3.5 Turbo was chosen for generating coherent and contextually relevant responses. It was integrated using the OpenAI API to handle user queries effectively.

## Search Functionality
The search workflow involves:

Query Embedding: An embedding is generated for each user query.

Cosine Similarity Calculation: Cosine similarity is calculated between the query embedding and course embeddings to identify the most relevant courses.
Results Display: The top 5 results are returned based on similarity scores.
## Database Management
ChromaDB manages the course embeddings, storing them in a collection for easy querying. Collections are refreshed with each update to ensure data consistency.

## User Interface
The UI, built with Gradio, provides a straightforward interface where users can enter search queries and receive a list of relevant courses, accompanied by a generated response from the GPT-3.5.

## Deployment
The Smart Search Tool is deployed on Hugging Face Spaces and accessible here:
Smart Search Tool on Hugging Face Spaces



https://huggingface.co/spaces/aditya-ml/smart_Search_tools

## Secrets and Local Setup

 - Create a local `.env` file from `.env.example` and add your OpenAI key:

```powershell
copy .\Smart_Search_tool\.env.example .\Smart_Search_tool\.env
# then edit the file and set OPENAI_API_KEY
```

 - Alternatively create a `secrets.json` (do not commit this file) in the `Smart_Search_tool` directory:

```json
{
	"OPENAI_API_KEY": "sk-..."
}
```

 - The FastAPI server reads `OPENAI_API_KEY` from the environment, `.env` (via python-dotenv), or `secrets.json`.

## Running locally

1. Install dependencies:

```powershell
python -m pip install -r .\Smart_Search_tool\requirements.txt
```

2. Start the FastAPI backend:

```powershell
uvicorn Smart_Search_tool.fastapi_backend:app --reload
```

3. Run the Gradio frontend (in a separate shell):

```powershell
python .\Smart_Search_tool\app.py
```

## Docker (optional)

Build the Docker image from the repository root and run it locally:

```powershell
docker build -t smart_search_tool -f .\Smart_Search_tool\Dockerfile .
docker run --rm -p 8000:8000 -p 7861:7861 --env OPENAI_API_KEY="sk-..." -v ${PWD}:/app smart_search_tool
```

Or use docker-compose (reads OPENAI_API_KEY from your environment):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
docker-compose -f .\Smart_Search_tool\docker-compose.yml up --build
```

Notes:
- The container runs both the FastAPI backend (port 8000) and the Gradio frontend (port 7861). The container mounts the repository to `/app`, so changes are visible inside the container.
- Do not commit secrets; use `.env` or `secrets.json` locally and list them in `.gitignore` (already added).

