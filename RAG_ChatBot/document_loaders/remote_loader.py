import requests
import os

from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from local_loader import get_document_text
from langchain_community.document_loaders import OnlinePDFLoader


# if you want it locally, you can use:
CONTENT_DIR = os.path.dirname(__file__)


# an alternative if you want it in /tmp or equivalent.
# CONTENT_DIR = tempfile.gettempdir()

def load_web_page(page_url):
    loader = WebBaseLoader(page_url)
    data = loader.load()
    return data


def load_online_pdf(pdf_url):
    loader = OnlinePDFLoader(pdf_url)
    data = loader.load()
    return data


def filename_from_url(url):
    filename = url.split("/")[-1]
    return filename


def download_file(url, filename=None):
    response = requests.get(url)
    if not filename:
        filename = filename_from_url(url)

    full_path = os.path.join(CONTENT_DIR, filename)

    with open(full_path, mode="wb") as f:
        f.write(response.content)
        download_path = os.path.realpath(f.name)
    print(f"Downloaded file {filename} to {download_path}")
    return download_path


def get_wiki_docs(query, load_max_docs=2):
    wiki_loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)
    docs = wiki_loader.load()
    for d in docs:
        print(d.metadata["title"])
    return docs