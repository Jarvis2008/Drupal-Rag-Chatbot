import os
from dotenv import load_dotenv
from groq import Groq
#from embedding import text_embedding
from neo4j_graph import neo4j_vector_search
from chroma_vectorstore_query import chroma_query_results


load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
#embed=text_embedding()
#question=input("Enter the question: ")
#question_embed=embed(question)
def graph_context(question_embed,k=4):
    query_out=neo4j_vector_search(question_embed,k)
    matched_info=""
    sources={'author_name':[],'author_id':[],'url':[]}
    for i in query_out:
        matched_info+=i['text']
        sources['author_name'].append(i['author_name'])
        sources['author_id'].append(i['author_id'])
        sources['url'].append(i['url'])

    context=f" Information {matched_info} and the sources:{sources}"
    return context
def chroma_context(question_embed,k=4):
    query_out=chroma_query_results(question_embed,k)
    matched_info=query_out['documents'][0][0]
    sources=query_out['metadatas']
    context=f"Information {matched_info} and the sources: {sources}"

    return context

def groq_output(question,context):
    chat_completion = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content":f""" Instructions:
                Your name is Jarvis. You are an AI ChatBot which has been built to help
                users on the Case studies available on the drupal.org website.
                Be helpful and answer quesitons concisely. If you don't know the
            answer, say 'I don't know'.
            Utilize the Context provided of the drupal casestudies, if any for accurate and specific information.
            Don't need to specify "According to context". Just answer question
            based on the information available to you.
            Incorporate your preexisting knowledge to enhance the depth and
            relevance of your response.
            Cite your sources.
            Context:{context}"""}
            ,

            {
            "role": "user",
            "content":f"{question}",
            }
            ,
            {
            "role": "assistant",
            "content": ""}
            ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

