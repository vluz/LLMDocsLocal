import re
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All


embeddings_model_name = "all-MiniLM-L6-v2"
model_path = "models/ggml-mpt-7b-instruct.bin"
pdf_folder_path = 'docs'
os.system('cls')
print("Loading Documents...")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings(), text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)
print("Done.")
llm = GPT4All(model=model_path, n_ctx=1000, backend='mpt', verbose=False)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(), return_source_documents=True)
os.system('cls')
print("\n\n\n")
print(" __    __    __  __    ____  _____  ___  ___    __    _____  ___    __    __   ")
print("(  )  (  )  (  \/  )  (  _ \(  _  )/ __)/ __)  (  )  (  _  )/ __)  /__\  (  )")
print(" )(__  )(__  )    (    )(_) ))(_)(( (__ \__ \   )(__  )(_)(( (__  /(__)\  )(__ ")
print("(____)(____)(_/\/\_)  (____/(_____)\___)(___/  (____)(_____)\___)(__)(__)(____)")

while True:
    prompt = input("\nPrompt: ")
    res = qa(prompt)
    answer = res['result']
    docs = res['source_documents']
    print("\nAnswer: ")
    print(answer)
    print("\n----------------------")
    for document in docs:
        texto = re.sub('[^A-Za-z0-9 ]+', '', document.page_content)
        print("\n" + document.metadata["source"] + ' -> ' + texto)
    print("\n######################")