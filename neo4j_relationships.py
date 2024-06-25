from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import GraphCypherQAChain
from database import preprocess_data
#from embedding import text_embedding
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

kg=Neo4jGraph(NEO4J_URI,NEO4J_USER,NEO4J_PASSWORD,NEO4J_DATABASE)
data=preprocess_data()


merge_author_nodes="""
MERGE(f:Author {author_name: $authorParam.author_name})
    ON CREATE SET
        f.author_id=$authorParam.author_id
    RETURN f
    """

merge_title_node="""
MERGE(f:Title {title: $titleParam.title})
    ON CREATE SET
        f.author_name=$titleParam.author_name,
        f.author_id=$titleParam.author_id,
        f.url=$titleParam.url,
        f.nid=$titleParam.nid,
        f.vid=$titleParam.vid
    RETURN f
    """
kg.query("""
         CREATE CONSTRAINT unique_author IF NOT EXISTS
         FOR (c:Author) Require c.author_name IS UNIQUE
         """)
kg.query("""
         CREATE CONSTRAINT unique_title IF NOT EXISTS
         FOR (c:Title) Require c.title IS UNIQUE
         """)
for chunk in data:
    kg.query(merge_author_nodes,params={'authorParam':chunk})
    kg.query(merge_title_node,params={'titleParam':chunk})


print(kg.query("SHOW INDEXES"))

titles=[]
for chunk in data:
    if chunk['title'] not in titles:
        titles.append(chunk['title'])

next_relation_cypher="""
MATCH (f:Chunk)
WHERE f.title=$title
WITH collect(f) as section_chunk_list
CALL apoc.nodes.link(
section_chunk_list,
"NEXT",
{avoidDuplicates:true}
)
RETURN section_chunk_list
"""
for title in titles:
    kg.query(next_relation_cypher,params={'title':title})

author_relation_cypher="""
MATCH (c:Chunk), (a:Author)
WHERE  c.author_name=a.author_name
MERGE (c)-[newRelationship:AUTHORED_BY]->(a)
"""
title_relation_cypher="""
MATCH (c:Chunk), (t:Title)
WHERE c.title=t.title
MERGE (c)-[newRelationship:HAS_TITLE]->(t)
"""
kg.query(author_relation_cypher)
kg.query(title_relation_cypher)
kg.refresh_schema()
print(kg.schema)
