import gradio as gr
from embedding import text_embedding
from groq_model import graph_context,chroma_context,groq_output
embed=text_embedding()
message_history=[]
graph_message_history=[]
chroma_message_history=[]
def  change_context(radio):
    if radio=="Neo4j context":
        return graph_message_history,gr.update(visible=True)
    elif radio=="Chromadb context":
        return chroma_message_history,gr.update(visible=True)
    else:
        return message_history,gr.update(visible=False)
def submit_button(message,radio,slider):
    question=embed(message)
    context=""
    if radio=="Neo4j context":
        context=graph_context(question,slider)
        out=groq_output(message,context)
        graph_message_history.append((message,out))
        return gr.update(value=graph_message_history),gr.update(value="")
    elif radio=="Chromadb context":
        context=chroma_context(question,slider)
        out=groq_output(message,context)
        chroma_message_history.append((message,out))
        return gr.update(value=chroma_message_history),gr.update(value="")
    else:
        context=context
        out=groq_output(message,context)
        message_history.append((message,out))
        return gr.update(value=message_history),gr.update(value="")

def clear_history(radio):
    if radio=="Neo4j context":
        return graph_message_history.clear()
    elif radio== "Chromadb context":
        return chroma_message_history.clear()
    else:
        return message_history.clear()
with gr.Blocks() as demo:
    radio=gr.Radio(
        ["No Context","Neo4j context","Chromadb context"],label="Which method do you want to use for Chat output?"
    )
    slider=gr.Slider(1,10,value=4,step=1,label="top k",info="Choose no of neighbours for context window",visible=False)
    chat=gr.Chatbot()
    msg=gr.Textbox(label="What is your question?")
    submit=gr.Button("Submit")
    radio.change(fn=change_context,inputs=[radio],outputs=[chat,slider])
    submit.click(submit_button,inputs=[msg,radio,slider],outputs=[chat,msg])
    clear=gr.ClearButton()
    clear.click(clear_history,inputs=[radio],outputs=[chat])
if __name__=="__main__":
    demo.launch(share=True)
