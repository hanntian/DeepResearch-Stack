import gradio as gr
import requests

API_URL = "http://localhost:8000/ask"


def ask_question(query: str):
    if not query.strip():
        return "Please enter a question.", ""

    response = requests.post(
        API_URL,
        json={"query": query, "top_k": 5},
        timeout=60,
    )

    if response.status_code != 200:
        return f"Error: {response.text}", ""

    data = response.json()

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    source_text = ""
    for idx, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        content = source.get("content", "")

        source_text += f"### Source {idx}: {title}\n"
        if url:
            source_text += f"{url}\n\n"
        source_text += f"{content[:800]}\n\n---\n\n"

    return answer, source_text


with gr.Blocks(title="DeepResearch Stack") as demo:
    gr.Markdown("# DeepResearch Stack")
    gr.Markdown( "Ask research questions and get most recent answers with sources.")

    query_input = gr.Textbox(
        label="Question",
        placeholder="Why is vLLM faster than HuggingFace inference?",
        lines=3,
    )

    submit_btn = gr.Button("Ask")

    answer_output = gr.Markdown(label="Answer")
    sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=ask_question,
        inputs=query_input,
        outputs=[answer_output, sources_output],
    )

if __name__ == "__main__":
    demo.launch()