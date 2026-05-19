import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

import arxiv
import requests
from bs4 import BeautifulSoup


OUTPUT_PATH = Path("data/raw_documents.jsonl")


def build_source_id(source, url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def save_jsonl(documents, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def ingest_arxiv(query="vLLM OR RAG OR Transformers", max_results=20):
    documents = []
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for result in client.results(search):
        documents.append(
            {
                "source_id": build_source_id("arxiv", result.entry_id),
                "source": "arxiv",
                "date": result.published.strftime("%Y-%m-%d"),
                "title": result.title,
                "content": result.summary,
                "url": result.entry_id,
                
            }
        )

    return documents


def ingest_web_page(url, source_name):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    content = " ".join(soup.get_text(separator=" ").split())

    return {
        "source_id": build_source_id(source_name, url),
        "source": source_name,
        "date": get_anthropic_blog_date(response.text),
        "title": title,
        "content": content,
        "url": url,
        
    }


def get_anthropic_blog_date(html):
    match = re.search(r'\\"publishedOn\\":\\"(\d{4}-\d{2}-\d{2})\\"', html)
    if match:
        return match.group(1)

    return datetime.today().strftime("%Y-%m-%d")


def ingest_anthropic_blog(urls=["https://www.anthropic.com/engineering"], max_entries=5):
    documents = []

    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            page_urls = [
                urljoin(url, anchor["href"]).split("?", 1)[0].split("#", 1)[0]
                for anchor in soup.find_all(
                    "a",
                    href=lambda href: href and href.startswith("/engineering/"),
                )[: max_entries - len(documents)]
            ]

            for page_url in page_urls:
                documents.append(ingest_web_page(page_url, "anthropic_engineering_blog"))

                if len(documents) >= max_entries:
                    return documents
        except Exception as error:
            print(f"Failed to ingest {url}: {error}")

    return documents


def main():
    documents = []

    print("Ingesting arXiv documents...")
    documents.extend(ingest_arxiv())

    print("Ingesting Anthropic Engineering blog documents...")
    documents.extend(ingest_anthropic_blog())

    save_jsonl(documents, OUTPUT_PATH)

    print(f"Saved {len(documents)} documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
