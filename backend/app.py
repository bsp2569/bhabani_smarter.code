import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)
CORS(app)  

chroma_client = chromadb.Client()
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def extract_elements(soup):
    """
    Walk the DOM and extract elements with meaningful text.
    We keep:
      - text: plain text used for embeddings
      - html: HTML snippet we’ll show in the UI
      - tag: tag name (optional, for debugging/UI)
    """
    body = soup.body or soup

    blacklist = {
        "script", "style", "noscript", "header", "footer", "nav",
        "svg", "meta", "link", "iframe"
    }

    elements = []

    for tag in body.find_all(recursive=True):
        if tag.name in blacklist:
            continue

        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        if len(text.split()) < 5:
            continue

        elements.append({
            "text": text,
            "html": str(tag),
            "tag": tag.name,
        })

    return elements


def chunk_elements_by_tokens(elements, max_tokens=500):
    """
    Take the extracted elements and merge them into chunks
    of at most `max_tokens` (simple whitespace tokenization).
    Each chunk combines text + HTML from multiple elements.
    """
    chunks = []
    current_tokens = []
    current_html = []
    current_tag = "div" 
    token_count = 0

    for el in elements:
        text = el["text"]
        html = el["html"]
        tokens = text.split()

        if not tokens:
            continue

        if token_count + len(tokens) > max_tokens and current_tokens:
            chunks.append({
                "text": " ".join(current_tokens),
                "html": "".join(current_html),
                "tag": current_tag,
            })
            current_tokens = []
            current_html = []
            token_count = 0

        current_tokens.extend(tokens)
        current_html.append(html)
        token_count += len(tokens)

    # Flush the last chunk
    if current_tokens:
        chunks.append({
            "text": " ".join(current_tokens),
            "html": "".join(current_html),
            "tag": current_tag,
        })

    return chunks


@app.route("/search", methods=["POST"])
def search_content():
    data = request.json
    url = data.get("url")
    query_text = data.get("query")

    if not url or not query_text:
        return jsonify({"error": "URL and Query are required"}), 400

    try:
        #fetch and parse the webpage
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        #clean unwanted tags
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        #Extract DOM elements
        raw_elements = extract_elements(soup)

        
        elements = chunk_elements_by_tokens(raw_elements, max_tokens=500)

        if not elements:
            return jsonify({"results": []})

        texts = [el["text"] for el in elements]   # for embeddings
        html_chunks = [el["html"] for el in elements]

        #Create temporary collection in ChromaDB
        collection_name = f"search_{uuid.uuid4().hex}"
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embed_fn,
        )

        ids = [str(i) for i in range(len(texts))]
        metadatas = [
            {
                "source": url,
                "index": i,
                "tag": elements[i]["tag"],
                "html": html_chunks[i],
            }
            for i in range(len(texts))
        ]

        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )

        #semantic search
        results = collection.query(
            query_texts=[query_text],
            n_results=min(10, len(texts)),
        )

        chroma_client.delete_collection(collection_name)

        #frontend output formatting
        formatted_results = []
        if results and "documents" in results:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0] if "metadatas" in results else [{}] * len(docs)
            distances = results.get("distances", [[]])[0] if "distances" in results else [0] * len(docs)

            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                formatted_results.append({
                    "text": doc,
                    "html": meta.get("html", ""),
                    "tag": meta.get("tag", ""),
                    "score": distances[i],
                })

        return jsonify({"results": formatted_results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Server running on port 5000...")
    app.run(debug=True, port=5000)
