import pandas as pd
import numpy as np
import torch
from transformers.modeling_utils import init_empty_weights
from sentence_transformers import SentenceTransformer
import faiss
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def encode_text_segments(df, model_name='all-MiniLM-L6-v2'):
    """
    Encode text segments using a Sentence-BERT model

    Args:
        df (pandas.DataFrame): DataFrame with text, start, and end timestamps
        model_name (str): Name of the sentence-transformers model to use

    Returns:
        tuple: (numpy array of embeddings, list of timestamp metadata)
    """
    model = SentenceTransformer(model_name)
    metadata = []
    text_segments = df['text'].tolist()
    embeddings = model.encode(text_segments,
                              show_progress_bar=True,
                              convert_to_numpy=True)
    for index, row in df.iterrows():
        metadata.append({
            'index': int(index),
            'start': float(row['start']),
            'end': float(row['end']),
            'text': row['text']
        })

    return embeddings, metadata

def create_faiss_index(embeddings, use_gpu=False):
    """
    Create a FAISS index for the embeddings

    Args:
        embeddings (numpy.ndarray): Numpy array of embeddings
        use_gpu (bool): Whether to use GPU for FAISS

    Returns:
        faiss.Index: FAISS index of embeddings
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("Using GPU FAISS index")
        except Exception as e:
            print(f"Failed to create GPU index: {e}")
    index.add(embeddings)

    return index

def process_transcript_data(csv_path, model_name='all-MiniLM-L6-v2', use_gpu=False):
    """
    Process transcript data, encode text, and create FAISS index

    Args:
        csv_path (str): Path to CSV file with transcript data
        model_name (str): Name of the sentence-transformers model to use
        use_gpu (bool): Whether to use GPU for FAISS

    Returns:
        tuple: (encoded embeddings, FAISS index, metadata)
    """
    df = pd.read_csv(csv_path)
    embeddings, metadata = encode_text_segments(df, model_name)
    faiss_index = create_faiss_index(embeddings, use_gpu)

    return embeddings, faiss_index, metadata, model_name

def embed_query(query, model_name='all-MiniLM-L6-v2'):
    """
    Embed a single query using the same model used for the corpus

    Args:
        query (str): Input query string to embed
        model_name (str): Name of the sentence-transformers model to use

    Returns:
        numpy.ndarray: Embedding vector for the query
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_numpy=True)
    if len(query_embedding.shape) == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    return query_embedding

def semantic_search(query, faiss_index, metadata, model_name, top_k=3):
    """
    Perform semantic search on the FAISS index

    Args:
        query (str): Input query to search
        faiss_index (faiss.Index): Precomputed FAISS index
        metadata (list): List of metadata dictionaries
        model_name (str): Name of the model used to create the index
        top_k (int, optional): Number of top results to return

    Returns:
        list: Top K most similar text segments with their metadata
    """
    query_embedding = embed_query(query, model_name)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        result_metadata = metadata[idx]
        result_metadata['distance'] = float(dist)
        results.append(result_metadata)

    return results

def save_data(faiss_index, metadata, embeddings, model_name, prefix='transcript'):
    """
    Save all data components to disk

    Args:
        faiss_index (faiss.Index): FAISS index
        metadata (list): Metadata list
        embeddings (numpy.ndarray): Embeddings array
        model_name (str): Model name used
        prefix (str): Prefix for saved files
    """

    with open(f'{prefix}_model.txt', 'w') as f:
        f.write(model_name)
    faiss.write_index(faiss_index, f'{prefix}_index.faiss')
    with open(f'{prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    np.save(f'{prefix}_embeddings.npy', embeddings)
    print(f"All data saved with prefix '{prefix}'")

def load_data(prefix='transcript'):
    with open(f'{prefix}_model.txt', 'r') as f:
        model_name = f.read().strip()
    faiss_index = faiss.read_index(f'{prefix}_index.faiss')
    with open(f'{prefix}_metadata.json', 'r') as f:
        metadata = json.load(f)
    return faiss_index, metadata, model_name
