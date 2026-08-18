"""
"""

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH     = "chroma_db"
COLLECTION  = "healthcare_misinfo"
EMBED_MODEL = "all-MiniLM-L6-v2"

KNOWLEDGE_BASE = [
    {
        "id": "who-001",
        "claim": "Drinking bleach or disinfectants can cure or prevent COVID-19.",
        "verdict": "misinformation",
        "source": "WHO COVID-19 myth-busters",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "rationale": "Ingesting disinfectants is extremely dangerous and has no evidence of treating COVID-19."
    },
    {
        "id": "who-002",
        "claim": "5G mobile networks spread or cause COVID-19.",
        "verdict": "misinformation",
        "source": "WHO COVID-19 myth-busters",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "rationale": "Viruses cannot travel on radio waves. COVID-19 spread in many countries without 5G networks."
    },
    {
        "id": "nhs-001",
        "claim": "COVID-19 vaccines alter your DNA.",
        "verdict": "misinformation",
        "source": "NHS COVID-19 vaccine facts",
        "url": "https://www.nhs.uk/conditions/coronavirus-covid-19/coronavirus-vaccination/",
        "rationale": "mRNA vaccines do not enter the cell nucleus and cannot alter human DNA."
    },
    {
        "id": "nhs-002",
        "claim": "The COVID-19 vaccine contains a microchip to track people.",
        "verdict": "misinformation",
        "source": "NHS COVID-19 vaccine facts",
        "url": "https://www.nhs.uk/conditions/coronavirus-covid-19/coronavirus-vaccination/",
        "rationale": "No microchip is present in any approved COVID-19 vaccine."
    },
    {
        "id": "cdc-001",
        "claim": "Antibiotics can treat viral infections such as colds, flu, or COVID-19.",
        "verdict": "misinformation",
        "source": "CDC Antibiotic Use",
        "url": "https://www.cdc.gov/antibiotic-use/index.html",
        "rationale": "Antibiotics only work against bacterial infections, not viruses."
    },
    {
        "id": "who-003",
        "claim": "Vaccines cause autism in children.",
        "verdict": "misinformation",
        "source": "WHO Vaccine safety",
        "url": "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-diseases",
        "rationale": "Extensive research involving millions of children has found no link between vaccines and autism."
    },
    {
        "id": "who-004",
        "claim": "Homeopathic remedies can cure cancer.",
        "verdict": "misinformation",
        "source": "WHO Traditional Medicine",
        "url": "https://www.who.int/health-topics/traditional-complementary-and-integrative-medicine",
        "rationale": "No credible clinical evidence supports homeopathy as a cancer treatment."
    },
    {
        "id": "cdc-002",
        "claim": "You can catch HIV from toilet seats or casual contact.",
        "verdict": "misinformation",
        "source": "CDC HIV Transmission",
        "url": "https://www.cdc.gov/hiv/basics/transmission.html",
        "rationale": "HIV is not transmitted through casual contact, toilet seats, air, or water."
    },
    {
        "id": "nhs-003",
        "claim": "Eating sugar directly causes diabetes.",
        "verdict": "misinformation",
        "source": "NHS Diabetes overview",
        "url": "https://www.nhs.uk/conditions/diabetes/",
        "rationale": "Type 1 is autoimmune. Type 2 is linked to obesity and lifestyle, not sugar alone."
    },
    {
        "id": "who-005",
        "claim": "Ivermectin cures or prevents COVID-19.",
        "verdict": "misinformation",
        "source": "WHO COVID-19 therapeutics",
        "url": "https://www.who.int/news-room/questions-and-answers/item/coronavirus-disease-covid-19-treatments",
        "rationale": "WHO does not recommend ivermectin for COVID-19 outside of clinical trials — evidence does not support its use."
    },
    {
        "id": "nhs-004",
        "claim": "You only use 10 percent of your brain.",
        "verdict": "misinformation",
        "source": "NHS Brain health",
        "url": "https://www.nhs.uk/",
        "rationale": "Brain imaging studies show virtually all parts of the brain have some function."
    },
    {
        "id": "cdc-003",
        "claim": "The flu vaccine gives you the flu.",
        "verdict": "misinformation",
        "source": "CDC Flu vaccination",
        "url": "https://www.cdc.gov/flu/prevent/misconceptions.htm",
        "rationale": "Flu vaccines are made from inactivated virus and cannot cause flu illness."
    },
    {
    "id": "who-airborne-001",
    "claim": "COVID-19 coronavirus is transmitted through the air and is airborne.",
    "verdict": "credible",
    "source": "WHO COVID-19 transmission update",
    "url": "https://www.who.int/news/item/30-04-2021-who-updates-on-covid-19-transmission",
    "rationale": "WHO confirmed in April 2021 that COVID-19 is airborne and can spread through aerosols in indoor settings, especially poorly ventilated spaces."
    },
    {
        "id": "who-airborne-002",
        "claim": "Coronavirus spreads through airborne aerosols in indoor environments.",
        "verdict": "credible",
        "source": "WHO COVID-19 transmission update",
        "url": "https://www.who.int/news/item/30-04-2021-who-updates-on-covid-19-transmission",
        "rationale": "Scientific evidence accumulated during the COVID-19 pandemic confirms airborne transmission as a primary route of spread."
    },
    {
        "id": "apple-cider-001",
        "claim": "Apple cider vinegar cures or significantly improves digestive problems and stomach issues.",
        "verdict": "misinformation",
        "source": "NHS Health A-Z",
        "url": "https://www.nhs.uk/conditions/",
        "rationale": "There is limited clinical evidence that apple cider vinegar significantly improves stomach conditions. It may cause harm in large quantities."
    },
    {
        "id": "apple-day-001",
        "claim": "Eating an apple every day prevents all illness and keeps you healthy.",
        "verdict": "misinformation",
        "source": "NHS Nutrition",
        "url": "https://www.nhs.uk/live-well/eat-well/",
        "rationale": "While apples are nutritious, no single food prevents all illness. The saying is a general wellness proverb, not a medical fact."
    }
]

def build_rag():
    print("Building RAG knowledge base...")
    encoder    = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    ids, embeddings, documents, metadatas = [], [], [], []

    for entry in tqdm(KNOWLEDGE_BASE, desc="Embedding entries"):
        emb = encoder.encode(entry["claim"]).tolist()
        ids.append(entry["id"])
        embeddings.append(emb)
        documents.append(entry["claim"])
        metadatas.append({
            "verdict":   entry["verdict"],
            "source":    entry["source"],
            "url":       entry["url"],
            "rationale": entry["rationale"],
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✓ RAG index built: {collection.count()} entries saved to {DB_PATH}/")
    print("Next step → run: python step3_run.py")

if __name__ == "__main__":
    build_rag()
