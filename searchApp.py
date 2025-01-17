import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

elastic_password = st.secrets.get("ELASTIC_PASSWORD")
if not elastic_password:
    st.error("Elastic password is not set in secrets.")
    st.stop()

try:

    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", elastic_password),
        ca_certs=st.secrets["CA_CERT_PATH"],
    )

    if es.ping():
        st.success("Successfully connected to ElasticSearch!")
    else:
        st.error("Failed to connect to Elasticsearch.")
        st.stop()
except ConnectionError as e:
    st.error(f"Connection Error: {e}")
    st.stop()


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
                            st.warning(f"Error displaying product name: {e}")

                        try:
                            st.write(f"Description: {result['_source']['Description']}")
                        except Exception as e:
                            st.warning(f"Error displaying description: {e}")
                        st.divider()


if __name__ == "__main__":
    main()
