import gradio as gr
import os
import socket
import requests
from typing import Optional, Tuple

# Backend API base URL (can be overridden via env var API_URL)
API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')
REQUEST_TIMEOUT = 10


def format_results_md(results: list) -> str:
    if not results:
        return "**No results found.**"
    parts = []
    for r in results:
        title = r.get('title', '')
        combined = r.get('combined_text', '')
        # Present title and description clearly; omit relevance score
        # Truncate long descriptions to keep UI tidy
        max_len = 800
        desc = combined.strip()
        if len(desc) > max_len:
            desc = desc[:max_len].rsplit(' ', 1)[0] + '...'
        # Use header for title and a short paragraph for description
        part = f"### {title}\n\n{desc}\n\n---"
        parts.append(part)
    return "\n\n".join(parts)


def search_and_generate_response(query: str) -> Tuple[str, str]:
    if not query or not query.strip():
        return ("", "Please enter a query.")

    # Call the FastAPI /search endpoint
    try:
        resp = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": 5},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return ("", f"Error contacting backend search API: {e}")

    results = data.get('results', [])
    results_md = format_results_md(results)

    # Call the backend generate endpoint (may require OPENAI_API_KEY set on server)
    try:
        gen_resp = requests.post(
            f"{API_URL}/generate",
            json={"prompt": query, "max_tokens": 256},
            timeout=REQUEST_TIMEOUT,
        )
        gen_resp.raise_for_status()
        gen_data = gen_resp.json()
        generated = gen_data.get('generated', '')
    except requests.RequestException as e:
        generated = f"(Generation unavailable: {e})"

    generated_md = f"## Generated Response\n\n{generated}"

    return (results_md, generated_md)


with gr.Blocks() as demo:
    gr.Markdown("# Analytics Vidhya Courses — Smart Search")
    gr.Markdown("Enter a query to search course content and get a generated summary.")

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(lines=2, placeholder="Type your query here...", elem_id="query")
            submit = gr.Button("Search")
            examples = gr.Examples(
                examples=["machine learning basics", "data engineering course for beginners", "deep learning curriculum"],
                inputs=[query]
            )

        with gr.Column(scale=3):
            results_md = gr.Markdown("", elem_id="results")
            gen_md = gr.Markdown("", elem_id="generated")

    submit.click(fn=search_and_generate_response, inputs=query, outputs=[results_md, gen_md])

    gr.Markdown("---\n*Make sure the FastAPI backend is running at the `API_URL` environment variable (default http://127.0.0.1:8000).*)")


def _find_free_port(start: int = 7861, end: int = 7870) -> int:
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise OSError(f"No free ports in range {start}-{end}")


def _launch():
    env_port = os.getenv('GRADIO_PORT')
    if env_port:
        port = int(env_port)
    else:
        port = _find_free_port(7861, 7870)
    demo.launch(share=True, server_port=port)


if __name__ == '__main__':
    _launch()
