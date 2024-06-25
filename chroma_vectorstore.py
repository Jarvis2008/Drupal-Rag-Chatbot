from database import preprocess_data
from embedding import text_embedding
import chromadb
from collections import Counter
import pickle
import os
data=preprocess_data()
data.pop(103)
data.pop(173)
data.pop(81)
data.pop(55)
custom_embed=text_embedding()
collection_name="drupal_vector"
client=chromadb.Client()
client=chromadb.PersistentClient(".")
#client.delete_collection(name=collection_name)
collection_exist=any(collection.name==collection_name for collection in client.list_collections())
if not collection_exist:
    collection=client.create_collection(name=collection_name)
else:
    collection=client.get_collection(name=collection_name)

embedding_file_exists=os.path.isfile('embeddings_chroma.pkl')
if not embedding_file_exists:
    embedded_data=[]
    for i in range(len(data)):
        embedded_data.append(custom_embed(data[i]['text']))
        print(f"Generated Embedding for node: {i+1}")
    with open('embeddings_chroma.pkl','wb') as f:
        pickle.dump(embedded_data,f)
else:
    with open('embeddings_chroma.pkl','rb') as f:
        embedded_data=pickle.load(f)

def convert_embedding(embeddings):
    return [list(map(float,embedding)) for embedding in embeddings]
documents=[]
metadatas=[]
ids=[]
embeddings=convert_embedding(embedded_data)
for i,chunk in enumerate(data):
    documents.append(chunk['text'])
    metadatas.append({'Title':chunk['title'],'author_name':chunk['author_name'],
                      'author_id':chunk['author_id'],'item':chunk['item'],'nid':chunk['nid']})
    ids.append(f"{i+1}")
if not all(isinstance(embedding,list) for embedding in embeddings):
    raise ValueError("Each embedding must be a list of numerical values.")
if len(embeddings) != len(documents):
    raise ValueError("Number of embeddings must match the number of documents.")
id_counts=Counter(documents)
duplicates=[id for i,(id,count) in enumerate(id_counts.items()) if count>1]

for i, text in enumerate(documents):
    if text in duplicates:
        print(data[i],i)
collection.add(documents=documents,metadatas=metadatas,ids=ids,embeddings=embeddings)
