import streamlit as st
import requests  # Or boto3 if using AWS credentials directly
from PIL import Image
from dotenv import load_dotenv
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers

load_dotenv()

session = boto3.session.Session()
service = 'aoss'
credentials = boto3.Session().get_credentials()
region_name = session.region_name
awsauth = AWSV4SignerAuth(credentials, region_name, service)
# Build the OpenSearch client
oss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# OpenSearch Connection Details (replace placeholders)
oss_endpoint = 'https://ylekgoldsyuhiknux16g.us-east-1.aoss.amazonaws.com'
index_name = 'titam-mm-index'
# Streamlit UI
st.title('Similar Item Finder')
query_prompt = st.text_input("Enter your query:")
if st.button('Search'):
    if query_prompt:
        # Fetch Similar Items from OpenSearch
        response = requests.post(oss_endpoint + '/' + index_name + '/_search', json={
            'query': {'match': {'content': query_prompt}}
        })
        # Handle Response and Display Results
        if response.status_code == 200:
            data = response.json()
            hits = data['hits']['hits']
            for hit in hits:
                image_url = hit['_source']['image_url']
                try:
                    image = Image.open(requests.get(image_url, stream=True).raw)
                    st.image(image, caption=hit['_source'].get('description', ''), use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load image from {image_url}: {e}")
        else:
            st.error(response.text)
            #st.error('OpenSearch query failed!')
    else:
        st.warning('Please enter a query.')