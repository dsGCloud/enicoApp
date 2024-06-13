import streamlit as st
from PIL import Image
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import pandas as pd
from io import BytesIO
from urllib.parse import urlparse
import boto3
import json
import base64
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
from sagemaker.s3 import S3Downloader

# Load the dataset (you might need to adapt this to your data source)
dataset = pd.read_csv("s3://amazon-berkeley-objects/images/metadata/images.csv.gz")  # Replace with your dataset path
# OpenSearch Connection Details (replace placeholders)
oss_endpoint = 'ylekgoldsyuhiknux16g.us-east-1.aoss.amazonaws.com'  # Replace with your actual endpoint
index_name = 'titam-mm-index' # Index name from the notebook
multimodal_embed_model = f'amazon.titan-embed-image-v1'
# Create Boto3 Session
# Define bedrock client
bedrock_client = boto3.client(
    "bedrock-runtime", 
    'us-east-1', 
    endpoint_url=f"https://bedrock-runtime.us-east-1.amazonaws.com"
)
session = boto3.Session(region_name='us-east-1')
credentials = session.get_credentials()
service = 'aoss'
s3_client = boto3.client('s3')
# AWS4Auth for signing requests
awsauth = AWS4Auth(credentials.access_key,
                   credentials.secret_key,
                   session.region_name,
                   service,
                   session_token=credentials.token)
# OpenSearch client
open_search_client = OpenSearch(
    hosts=[{'host': oss_endpoint, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)
image_meta = pd.read_csv("s3://amazon-berkeley-objects/images/metadata/images.csv.gz")
meta = pd.read_json("s3://amazon-berkeley-objects/listings/metadata/listings_0.json.gz", lines=True)
def func_(x):
    us_texts = [item["value"] for item in x if item["language_tag"] == "en_US"]
    return us_texts[0] if us_texts else None
meta = meta.assign(item_name_in_en_us=meta.item_name.apply(func_))
meta = meta[~meta.item_name_in_en_us.isna()][["item_id", "item_name_in_en_us", "main_image_id"]]
dataset = meta.merge(image_meta, left_on="main_image_id", right_on="image_id")
dataset = dataset.assign(img_full_path=f's3://amazon-berkeley-objects/images/small/' + dataset.path.astype(str))
def get_titan_multimodal_embedding(
    image_path:str=None,  # maximum 2048 x 2048 pixels
    description:str=None, # English only and max input tokens 128
    dimension:int=1024,   # 1,024 (default), 384, 256
    model_id:str=multimodal_embed_model
):
    payload_body = {}
    embedding_config = {
        "embeddingConfig": { 
             "outputEmbeddingLength": dimension
         }
    }
    # You can specify either text or image or both
    if image_path:
        if image_path.startswith('s3'):
            s3 = boto3.client('s3')
            bucket_name, key = image_path.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            # Read the object's body
            body = obj['Body'].read()
            # Encode the body in base64
            base64_image = base64.b64encode(body).decode('utf-8')
            payload_body["inputImage"] = base64_image
        else:   
            with open(image_path, "rb") as image_file:
                input_image = base64.b64encode(image_file.read()).decode('utf8')
            payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"
    # print("\n".join(payload_body.keys()))

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}), 
        modelId=model_id,
        accept="application/json", 
        contentType="application/json"
    )

    return json.loads(response.get("body").read())
def find_similar_items_from_query(open_search_client, dataset, query_prompt: str, index_name: str, k: int = 5, num_results: int = 3, image_root_path: str = 's3://amazon-berkeley-objects/images/small/') -> []:
    text_embedding = get_titan_multimodal_embedding(description=query_prompt, dimension=1024)["embedding"]
    search_query = {
        "size": num_results,
        "query": {
            "knn": {
                "image_vector": {
                    "vector": text_embedding,
                    "k": k
                }
            }
        },
        "_source": True
    }
    response = open_search_client.search(
        index=index_name,
        body=search_query,
    )
    similar_items = []
    
    for hit in response["hits"]["hits"]:
        id_ = hit["_id"]
        item_id_ = hit["_source"]["item_id"]
        # image, item_name = get_image_from_item_id(item_id = id_, dataset = dataset )
        image, item_name = get_image_from_item_id_s3(item_id = item_id_, dataset = dataset,image_path = image_root_path)
        image.name_and_score = f'{hit["_score"]}:{item_name}'
        similar_items.append({
            'image': image,
            'item_name': item_name,
            'score': hit["_score"]
        })
        
    return similar_items
def get_image_from_item_id_s3(item_id, dataset, image_path,  return_image=True):

    item_idx = dataset.query(f"item_id == '{item_id}'").index[0]
    img_loc =  dataset.iloc[item_idx].img_full_path
    
    if img_loc.startswith('s3'):
        # download and store images locally 
        local_data_root = f'./data/images'
        local_file_name = img_loc.split('/')[-1]
 
        S3Downloader.download(img_loc, local_data_root)
 
    local_image_path = f"{local_data_root}/{local_file_name}"
    
    if return_image:
        img = Image.open(local_image_path)
        return img, dataset.iloc[item_idx].item_name_in_en_us
    else:
        return local_image_path, dataset.iloc[item_idx].item_name_in_en_us
def display_images(simiar_items, 
    columns=2, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):
 
    if not similar_items:
        st.warning("No images to display.")
        return 
 
    if len(similar_items) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]
    for item in similar_items:
        image = item['image']
        item_name = item['item_name']
        score = item['score']
        st.image(image, caption=f'{item_name} (Score: {score:.2f})', use_column_width=True)
# Streamlit UI
st.title('EniCommerce - Semantic Search Demo')
query_prompt = st.text_input("Hoy, qué buscas para tu mascota?")
if st.button('Search'):
    if query_prompt:
        similar_items = find_similar_items_from_query(query_prompt=query_prompt, index_name=index_name, open_search_client=open_search_client, dataset=dataset)
        # Handle Response and Display Results
        if not similar_items:
            st.error('OpenSearch query did not return expected results')
        else:
            display_images(similar_items)
    else:
        st.warning('Please enter a query.')
