import os
from typing import List, Optional, Dict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
from bs4 import BeautifulSoup
import re
# Placeholder imports for future implementation
# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader

class RegBot:
    """
    Main class for the GA4GH Compliance Assistant.
    """
    load_dotenv()
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    ):
        self.api_key = api_key or os.getenv("HF_API_TOKEN")
        if not self.api_key:
            raise ValueError("HF_API_TOKEN not found")
        
        self.vector_db = None
        self.model_name = model_name
        self.load_llm()
        print("Initializing RegBot Core...")

    def load_llm(self):
        #Initialize the Hugging Face Inference API client.

        self.client = InferenceClient(
            model=self.model_name,
            token=self.api_key
        )

    # def ingest_policy_documents(self, url: str) -> List[Dict[str,str]]:
    #     """
    #     Phase 1: Load GA4GH Framework PDF and convert to embeddings.
    #     TODO: Implement PyPDFLoader and recursive character splitting.
    #     """
    #     """

    #     For this PoC we will scrape from the web directly, but in the future we will implement PDF ingestion and embedding storage in ChromaDB.
    #     """
    #     print(f"Loading policy document from: {url}")
    #     response =requests.get(url)
    #     if response.status_code != 200:
    #         print(f"Failed to fetch document: {response.status_code}")
    #         return []
    #     clauses = []
    #     soup = BeautifulSoup(response.text,"html.parser")
    #     headings = soup.find_all(['h1','h2','h3','h4'])

            
    #     for h in headings:
    #         text = h.get_text(strip=True)
    #         if not text:
    #             continue
    #         match = re.match(r'^(\d+(\.\d+)*)', text)
    #         if not match:
    #             continue
    #         clause_number = match.group(1)
    #         clause_text = []
    #         for sib in h.next_elements:
    #             if hasattr(sib,'get_text'):
    #                 content = sib.get_text(strip=True)
    #                 if content:
    #                     clause_text.append(content)

    #         full_text = " ".join(clause_text)
    #         if not full_text:
    #             full_text=text

    #         clauses.append({
    #             "clause": clause_number,
    #              "text": full_text
    #              })
    #     print(f"Ingested {len(clauses)} clauses from framework")
    #     return clauses

    def ingest_policy_documents(self, url: str) -> list:
        print(f"Loading policy document from: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch document: {response.status_code}")
            return []

        clauses = []
        soup = BeautifulSoup(response.text, "html.parser")
        headings = soup.find_all(['h1','h2','h3','h4','strong'])

        for idx, h in enumerate(headings):
            text = h.get_text(strip=True)
            match = re.match(r'^(\d+(\.\d+)*)', text)
            if not match:
                continue
            clause_number = match.group(1)

            # Determine the text of the next numbered heading
            next_heading_text = None
            for nh in headings[idx+1:]:
                nh_text = nh.get_text(strip=True)
                if re.match(r'^\d+(\.\d+)*', nh_text):
                    next_heading_text = nh_text
                    break

            # Collect all elements until we hit a heading with next_heading_text
            clause_text = []
            for elem in h.find_all_next():
                if hasattr(elem, 'get_text'):
                    content = elem.get_text(strip=True)
                    if content == next_heading_text:
                        break
                    if content:
                        clause_text.append(content)

            full_text = " ".join(clause_text).strip()
            if not full_text:
                full_text = text

            clauses.append({"clause": clause_number, "text": full_text})

        print(f"Ingested {len(clauses)} clauses from framework")
        return clauses

    def retrieve_relevant_clauses(self, user_query: str) -> List[str]:
        """
        Phase 2: RAG Implementation.
        Search vector DB for clauses relevant to the user's consent form.
        """
        print("Retrieving regulatory context...")
        return ["Clause 4.1: Data Sharing", "Clause 2.3: Patient Consent"]

    def check_compliance(self, user_consent_form: str) -> dict:
        """
        Phase 3: LLM Analysis.
        Compares user input against retrieved GA4GH clauses.
        """
        print("Analyzing compliance gap...")
        # Placeholder for LLM Chain
        return {
            "status": "Non-Compliant",
            "missing_elements": ["Data Use Limitation", "Cloud Storage Provision"],
            "suggested_fix": "Add specific clause regarding secondary use of data."
        }

if __name__ == "__main__":
    # Entry point for testing the pipeline
    bot = RegBot()
    clauses = bot.ingest_policy_documents("https://www.ga4gh.org/framework/")
    for c in clauses[:1]:
        print(f"Clause: {c['clause']}")
        print(f"Text preview: {c['text'][:]}...\n")

