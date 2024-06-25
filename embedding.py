from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
#from chromadb.api.types import EmbeddingFunction
warnings.filterwarnings('ignore')
model=SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5',cache_folder='./embedding_model',trust_remote_code=True)
sentence=["Hello My name is Jaikirat Singh"]
class text_embedding():
    def __init__(self):
        self.model=model

    def embed_query(self,texts):
        embeddings=self.model.encode(texts)
        return embeddings.tolist()
    def __call__(self,texts):
        embeddings=self.model.encode(texts)
        return embeddings
    def embed_documents(self,documents):
        embeddings=[]
        for text in documents:
            response=self.embed_query(text)
            if not isinstance(response,dict):
                response=response.dict()
            embeddings.extend(r["embedding"] for r in repsonse["data"])
        return embeddings
print("Successfully loaded embeddings")
