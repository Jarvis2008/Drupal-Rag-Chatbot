import chromadb

client=chromadb.Client()
client=chromadb.PersistentClient(".")
collection=client.get_collection('drupal_vector')
def convert_embedding(embeddings):
    return [list(map(float,embedding)) for embedding in embeddings]
def chroma_query_results(question,k=3):
    results=collection.query(
        query_embeddings=convert_embedding([question]),
        n_results=k

    )
    return results
