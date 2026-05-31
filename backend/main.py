# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# import pandas as pd
# import numpy as np

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity


# # =====================================================
# # APP SETUP
# # =====================================================

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # =====================================================
# # LOAD DATA
# # =====================================================

# # # ML / Categorized dataset
# # df = pd.read_json(
# #     "data/categorized_cases.json"
# # )
# # df = df.replace(
# #     {np.nan: None}
# # )

# # # Full original cases dataset
# # full_cases_df = pd.read_json(
# #     "data/output.json"
# # )
# # full_cases_df = full_cases_df.replace(
# #     {np.nan: None}
# # )

# # # Embeddings
# # embeddings = np.load(
# #     "data/description_embeddings.npy"
# # )

# # # Embedding model
# # # model = SentenceTransformer(
# # #     "all-MiniLM-L6-v2"
# # # )

# df = None
# full_cases_df = None
# embeddings = None
# model = None


# # =====================================================
# # HOME ROUTE
# # =====================================================

# @app.get("/")
# def home():

#     return {
#         "message": "Missing Person Similarity API Running"
#     }


# # =====================================================
# # BASIC CASE ROUTES
# # =====================================================

# # @app.get("/cases")
# # def get_cases():

# #     return df.to_dict(
# #         orient="records"
# #     )


# # @app.get("/cases/{case_id}")
# # def get_case(case_id: int):

# #     case = df[df["id"] == case_id]

# #     if case.empty:
# #         return {
# #             "error": "Case not found"
# #         }

# #     return case.iloc[0].to_dict()


# # =====================================================
# # FULL CASE ROUTES
# # =====================================================

# # @app.get("/full-cases")
# # def get_full_cases():

# #     return full_cases_df.to_dict(
# #         orient="records"
# #     )


# # @app.get("/full-cases/{case_id}")
# # def get_full_case(case_id: int):

# #     case = full_cases_df[
# #         full_cases_df["id"] == case_id
# #     ]

# #     if case.empty:

# #         return {
# #             "error": "Case not found"
# #         }

# #     return case.iloc[0].to_dict()


# # =====================================================
# # SIMILAR CASES
# # =====================================================

# # @app.get("/similar/{case_id}")
# # def find_similar_cases(
# #     case_id: int,
# #     top_n: int = 5
# # ):

# #     # Find row index
# #     case_index = df.index[
# #         df["id"] == case_id
# #     ][0]

# #     # Get embedding
# #     case_embedding = embeddings[
# #         case_index
# #     ].reshape(1, -1)

# #     # Similarities
# #     similarities = cosine_similarity(
# #         case_embedding,
# #         embeddings
# #     )[0]

# #     # Remove self
# #     similarities[case_index] = -1

# #     # Top matches
# #     top_indices = similarities.argsort()[
# #         -top_n:
# #     ][::-1]

# #     results = df.iloc[top_indices][[
# #         "id",
# #         "description",
# #         "topic_category"
# #     ]].copy()

# #     results["similarity_score"] = (
# #         similarities[top_indices]
# #     )

# #     return results.to_dict(
# #         orient="records"
# #     )


# # =====================================================
# # SEMANTIC SEARCH
# # =====================================================

# # @app.get("/search")
# # def semantic_search(
# #     query: str,
# #     top_n: int = 5
# # ):

# #     # Convert query to embedding
# #     query_embedding = model.encode(
# #         [query]
# #     )

# #     # Similarities
# #     similarities = cosine_similarity(
# #         query_embedding,
# #         embeddings
# #     )[0]

# #     # Top matches
# #     top_indices = similarities.argsort()[
# #         -top_n:
# #     ][::-1]

# #     results = df.iloc[top_indices][[
# #         "id",
# #         "description",
# #         "topic_category"
# #     ]].copy()

# #     results["similarity_score"] = (
# #         similarities[top_indices]
# #     )

# #     return results.to_dict(
# #         orient="records"
# #     )

# @app.get("/search")
# def semantic_search():

#     return {
#         "message": "Search temporarily disabled"
#     }

# @app.get("/cases")
# def get_cases():
#     return {"message": "test"}

# @app.get("/full-cases")
# def get_full_cases():
#     return {"message": "test"}


from fastapi import FastAPI
import pandas as pd
import numpy as np

app = FastAPI()

df = pd.read_json(
    "data/categorized_cases.json"
)

@app.get("/")
def home():
    return {
        "message": "Backend working"
    }

embeddings = np.load(
    "data/description_embeddings.npy"
)