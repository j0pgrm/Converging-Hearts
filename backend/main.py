import os 
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_json(
    "data/categorized_cases.json"
)

df = df.replace(
    {np.nan: None}
)

full_cases_df = pd.read_json(
    "data/output.json"
)

full_cases_df = full_cases_df.replace(
    {np.nan: None}
)

embeddings = np.load(
    "data/description_embeddings.npy"
)


# 


HF_TOKEN = os.getenv("HF_TOKEN")

HF_API_URL = (
    "https://router.huggingface.co/hf-inference/"
    "models/sentence-transformers/"
    "all-MiniLM-L6-v2"
)


def get_query_embedding(text):

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={
            "inputs": text
        },
        timeout=30
    )

    return {
        "status": response.status_code,
        "text": response.text
    }


# 



@app.get("/")
def home():

    return {
        "message": "Missing Person Similarity API Running"
    }

@app.get("/cases")
def get_cases():

    return df.to_dict(
        orient="records"
    )

@app.get("/cases/{case_id}")
def get_case(case_id: int):

    case = df[
        df["id"] == case_id
    ]

    if case.empty:

        return {
            "error": "Case not found"
        }

    return case.iloc[0].to_dict()

@app.get("/full-cases")
def get_full_cases():

    return full_cases_df.to_dict(
        orient="records"
    )

@app.get("/full-cases/{case_id}")
def get_full_case(case_id: int):

    case = full_cases_df[
        full_cases_df["id"] == case_id
    ]

    if case.empty:

        return {
            "error": "Case not found"
        }

    return case.iloc[0].to_dict()

@app.get("/similar/{case_id}")
def find_similar_cases(
    case_id: int,
    top_n: int = 5
):

    matching_rows = df.index[
        df["id"] == case_id
    ]

    if len(matching_rows) == 0:

        return {
            "error": "Case not found"
        }

    case_index = matching_rows[0]

    case_embedding = embeddings[
        case_index
    ].reshape(1, -1)

    similarities = cosine_similarity(
        case_embedding,
        embeddings
    )[0]

    similarities[case_index] = -1

    top_indices = similarities.argsort()[
        -top_n:
    ][::-1]

    results = df.iloc[top_indices][[
        "id",
        "description",
        "topic_category"
    ]].copy()

    results["similarity_score"] = (
        similarities[top_indices]
    )

    return results.to_dict(
        orient="records"
    )



# @app.get("/search")
# def semantic_search(
#     query: str,
#     top_n: int = 20
# ):

#     query_embedding = np.array(
#         get_query_embedding(query)
#     ).reshape(1, -1)

#     similarities = cosine_similarity(
#         query_embedding,
#         embeddings
#     )[0]

#     top_indices = similarities.argsort()[
#         -top_n:
#     ][::-1]

#     results = df.iloc[top_indices][[
#         "id",
#         "description",
#         "topic_category"
#     ]].copy()

#     results["similarity_score"] = (
#         similarities[top_indices]
#     )

#     return results.to_dict(
#         orient="records"
#     )

@app.get("/search")
def semantic_search(
    query: str,
    top_n: int = 20
):

    return get_query_embedding(query)