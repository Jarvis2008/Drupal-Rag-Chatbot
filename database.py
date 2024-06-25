#Importing the libraries
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import html2text
import warnings
warnings.filterwarnings("ignore")

# Importing the Data from the API.
url =r"https://www.drupal.org/api-d7/node.json?type=casestudy"
response=requests.get(url)
if response.status_code==200:
  data=response.json()['list']
  print("Successfully downloaded the JSON file.")
else:
  print("Unable to fetch data from the URL")

# Cleaning the Dataset

data[24]['author']={'uri':"",'id':"","resource":'user','name':'unnamed'}


# Setting up a text splitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=150,
                                             length_function=len)
#Html to text
def html_to_text(html_text):
    h=html2text.HTML2Text()
    h.ignorelinks=True
    text=h.handle(html_text)
    text=''.join(text.splitlines())
    return text
def preprocess_data():
    chunk_metadata=[]
    items=["body","field_developed","field_goals","field_overview"]
    # Processing the data
    for j in range(len(data)):
        for i in items:
            if i in data[j] and isinstance(data[j][i], dict) and 'value' in data[j][i]:
                item_text = data[j][i]['value']
                item_text_chunk = text_splitter.split_text(item_text)
                chunk_seq_ids = 0
                title=data[j]['title']
                for chunk in item_text_chunk:
                    chunk_metadata.append({"text":html_to_text(chunk),
                          "item":i,
                          "chunkId":f'{title}-{i}-chunk-{chunk_seq_ids}',
                          "title":title,
                          "url":data[j]['url'],
                          'author_id':data[j]['author']['id'],
                          'author_name':data[j]['author']['name'],
                          'nid':data[j]['nid'],
                          'vid':data[j]['vid'],
                          })
                    chunk_seq_ids += 1
    return chunk_metadata

if __name__=="__main__":
    print(len(preprocess_data()),preprocess_data()[0:5])
