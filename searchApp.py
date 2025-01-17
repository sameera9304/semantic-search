import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os

indexName = "all_products"
elastic_password = os.getenv("ELASTIC_PASSWORD")
if not elastic_password:
    raise ValueError("Environment variable ELASTIC_PASSWORD is not set.")
try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", elastic_password),
        ca_certs="D:/elasticsearch-8.17.0-windows-x86_64/elasticsearch-8.17.0/config/certs/http_ca.crt",
    )

except ConnectionError as e:
    print("Connection Error:", e)

if es.ping():
    print("Succesfully connected to ElasticSearch!!")
else:
    print("Oops!! Can not connect to Elasticsearch!")


def search(input_keyword):
    model = SentenceTransformer("all-mpnet-base-v2")
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 500,
    }
    res = es.knn_search(
        index="all_products", knn=query, source=["ProductName", "Description"]
    )
    results = res["hits"]["hits"]

    return results


def main():
    st.title("Search Myntra Fashion Products")
    search_query = st.text_input("Enter your search query")

    if st.button("Search"):
        if search_query:
            results = search(search_query)
            st.subheader("Search Results")
            for result in results:
                with st.container():
                    if "_source" in result:
                        try:
                            st.header(f"{result['_source']['ProductName']}")
                        except Exception as e:
                            print(e)

                        try:
                            st.write(f"Description: {result['_source']['Description']}")
                        except Exception as e:
                            print(e)
                        st.divider()


if __name__ == "__main__":
    main()
