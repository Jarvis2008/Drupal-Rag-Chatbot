from langchain_community.graphs import Neo4jGraph
import os

NEO4J_URI = os.getenv("NEO4J_URI","bolt://neo4j:7687")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE="neo4j"
kg=Neo4jGraph(NEO4J_URI,NEO4J_USER,NEO4J_PASSWORD,NEO4J_DATABASE)
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
#print(kg.schema)

if __name__ == '__main__':
    from embedding import text_embedding
    embed=text_embedding()
    print(neo4j_vector_search(embed("What are some of the projects covered under Symnatec Connect")))
