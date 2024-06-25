
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import GraphCypherQAChain
#from database import preprocess_data
from embedding import text_embedding
import warnings
warnings.filterwarnings('ignore')
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

class CustomNeo4jVector(VectorStore):
    def __init__(self,url: str, username: str, password: str, database: str,
                    index_name:str,dimension:int,embedding:Embeddings, node_label: str = "Chunk",
                 embedding_property: str = "textEmbedding"):
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self.database = database
        self.embedding = embedding
        self.node_label = node_label
        self.embedding_property = embedding_property
        self.dimension=dimension
        self.index_name=index_name

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        embeddings = self.embedding.embed_documents(texts)

        with self.driver.session(database=self.database) as session:
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                metadata = metadatas[i] if metadatas else {}
                result = session.run(
                    f"""
                    CREATE (c:{self.node_label} {{text: $text}})
                    SET c += $metadata
                    WITH c
                    CALL db.create.setNodeVectorProperty(c, $embedding_property, $embedding)
                    RETURN id(c) AS id
                    """,
                    text=text,
                    embedding_property=self.embedding_property,
                    embedding=embedding,
                    metadata=metadata
                )
                results.append(str(result.single()['id']))
            return results

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = self.embedding.embed_query(query)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
                YIELD node, score
                RETURN node.text AS text, score, id(node) AS id
                """,
                index_name=self.index_name,
                k=k,
                query_embedding=query_embedding
            )
            return [{"text": record["text"], "score": record["score"], "id": str(record["id"])} for record in result]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        url: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        node_label: str = "Chunk",
        embedding_property: str = "textEmbedding",
        dimension: int = 1536,
        **kwargs: Any,
    ) -> "CustomNeo4jVector":
        instance = cls(url, username, password, database, embedding, node_label, embedding_property, dimension)
        instance.add_texts(texts, metadatas)
        return instance
    @classmethod
    def from_existing_graph(cls, url: str, username: str, password: str, database: str,
                            index_name:str,dimension:int,embedding: Embeddings, node_label: str = "Chunk",
                            embedding_property: str = "textEmbedding"):
        instance = cls(url, username, password, database,index_name,dimension,embedding, node_label, embedding_property)

      # print(f"index_name: {instance.index_name}, type: {type(instance.index_name)}")
       # print(f"node_label: {instance.node_label}, type: {type(instance.node_label)}")
       # print(f"embedding_property: {instance.embedding_property}, type: {type(instance.embedding_property)}")
       # print(f"dimension: {instance.dimension}, type: {type(instance.dimension)}")

        print(f"Using the existing vector index: {instance.index_name}")

        return instance

    def as_retriever(self, search_kwargs=None):
        from langchain.schema import BaseRetriever
        from langchain.schema.document import Document
        from pydantic import BaseModel,Field
        from typing import Dict,Any,List
        class CustomRetriever(BaseRetriever):
            vectorstore:Any=Field(...)
            search_kwargs:Dict[str,Any] = Field(default_factory=dict)

            class Config:
                arbitrary_types_allowed=True

            def get_relevant_documents(self, query: str) -> List[Document]:
                results = self.vectorstore.similarity_search(query, **self.search_kwargs)
                return [Document(page_content=result['text'], metadata={'id': result['id']}) for result in results]

            async def aget_relevant_documents(self, query: str) -> List[Document]:
                return self.get_relevant_documents(query)

        return CustomRetriever(vectorstore=self, search_kwargs=search_kwargs or {})
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE="neo4j"

VECTOR_INDEX_NAME="drupal_vector"
VECTOR_DIMENSION=1024
VECTOR_NODE_LABEL='Chunk'
VECTOR_SOURCE_PROPERTY='text'
VECTOR_EMBEDDING_PROPERTY='textEmbedding'

neo4j_vector_store=CustomNeo4jVector.from_existing_graph(
    embedding=text_embedding(),
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    dimension=VECTOR_DIMENSION,
    embedding_property=VECTOR_EMBEDDING_PROPERTY,
)
#print(neo4j_vector_store.similarity_search('What is symnatec connect'))
def neo4j_retriever():
    return neo4j_vector_store.as_retriever()

