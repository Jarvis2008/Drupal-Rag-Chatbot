from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import GraphCypherQAChain
from database import preprocess_data
from embedding import text_embedding
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
# Setting up neo4j parameters
NEO4J_URI = os.getenv("NEO4J_URI","bolt://neo4j:7687")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE="neo4j"

VECTOR_INDEX_NAME="drupal_vector"
VECTOR_DIMENSION=1024
VECTOR_NODE_LABEL='Chunk'
VECTOR_SOURCE_PROPERTY='text'
VECTOR_EMBEDDING_PROPERTY='textembedding'

text_embedding=text_embedding()
kg=Neo4jGraph(NEO4J_URI,NEO4J_USER,NEO4J_PASSWORD,NEO4J_DATABASE)
#kg.query(""" MATCH (a) -[r]-> () DELETE a,r """)
#kg.query(""" MATCH (a) DELETE a """)
# Importing the Preprocess data for knowledge Graphs
data=preprocess_data()

merge_chunk_nodes= """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET
        mergedChunk.author_name=$chunkParam.author_name,
        mergedChunk.author_id=$chunkParam.author_id,
        mergedChunk.title=$chunkParam.title,
        mergedChunk.url=$chunkParam.url,
        mergedChunk.nid=$chunkParam.nid,
        mergedChunk.vid=$chunkParam.vid,
        mergedChunk.item=$chunkParam.item,
        mergedChunk.text=$chunkParam.text
RETURN mergedChunk
"""
#a=kg.query(merge_chunk_nodes,params={'chunkParam':data[0]})

kg.query("""
         CREATE CONSTRAINT unique_chunk IF NOT EXISTS
         FOR (c:Chunk) Require c.chunkId IS UNIQUE
         """)

node_count=0
for chunk in data:
    kg.query(merge_chunk_nodes,params={'chunkParam':chunk})
    node_count+=1
print(f"Created {node_count} nodes ")




embedding_file_exists=os.path.isfile('embeddings_graph.pkl')
if not embedding_file_exists:
     embedded_data=[]
     for i in range(node_count):
         embedded_data.append({'chunkId':data[i]['chunkId'],"embedding":text_embedding(data[i]['text'])})
         print(f"Generated Embedding for node: {i+1}")
     with open('embeddings_graph.pkl','wb') as f:
         pickle.dump(embedded_data,f)
else:
    with open('embeddings_graph.pkl','rb') as f:
        embedded_data=pickle.load(f)
#print(kg.query("""MATCH (n:Chunk) RETURN n LIMIT 1"""))

kg.query("""
         CREATE VECTOR INDEX drupal_vector IF NOT EXISTS
         FOR (c:Chunk) ON (c.textEmbedding)
         OPTIONS { indexConfig: {
            `vector.dimensions`:1024,
            `vector.similarity_function`:'cosine'
         }}
         """)


kg.query("""
         UNWIND $embedded_data as row
         MATCH (n:Chunk) WHERE n.chunkId = row.chunkId
         CALL db.create.setNodeVectorProperty(n,"textEmbedding",row.embedding)
         """,params={'embedded_data':embedded_data})

kg.refresh_schema()
#print(kg.query(""" MATCH (n) RETURN n LIMIT 1 """))
import neo4j_relationships


def neo4j_vector_search(question,k:int=3):
    question_embed=question
    vector_search_query="""
    CALL db.index.vector.queryNodes('drupal_vector',$top_k,$question_embedding) yield  node,score
    RETURN score,node.text AS text,node.author_name AS author_name,
    node.author_id AS author_id,node.url AS url
    """
    similar=kg.query(vector_search_query,params={
        'question_embedding':question_embed,
        'top_k':k})

    return similar


