import re
import os
import gradio as gr
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All


embeddings_model_name = "all-MiniLM-L6-v2"
model_path = "models/ggml-mpt-7b-instruct.bin"
pdf_folder_path = 'docs'
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings(), text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)
llm = GPT4All(model=model_path, n_ctx=1000, backend='mpt', verbose=False)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(), return_source_documents=True)
examples = [["How much did EDP earn in the last quarter of 2022?"],]

def generate(prompt):
    res = qa(prompt)
    answer = res['result']
    docs = res['source_documents']
    output = answer + "\n"
    for document in docs:
        output += ("\n\n\n" + document.metadata["source"] + ' -> ' + document.page_content)
    print(output)
    return output

app = gr.Interface(
    fn=generate,
    inputs=gr.inputs.Textbox(label="Prompt"),
    outputs=gr.outputs.Textbox(label="Answer"),
    examples=examples
)

app.launch()

