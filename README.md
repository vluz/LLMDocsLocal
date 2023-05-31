# LLMDocsLocal
### Use a local LLM to read and provide answers on local files

Uses https://huggingface.co/mosaicml/mpt-7b-instruct, a model from
<br>
Mosaic ML https://mosaicml.com 


PDF documents in the `docs` folder are loaded into a Vectorstore

The model can then answer questions about the ducuments it has ingested.

There are two versions one CLI, and one based on Gradio

<hr>

(optional) You can create a virtual environment with:
```
python -m venv "venv"
venv\Scripts\activate
```

To install do:
```
git clone https://github.com/vluz/LLMDocsLocal.git
cd LLMDocsLocal
pip install -r requirements.txt
```

To run do:<br>
`python answers.py` for the cli version
<br>or
`python answersgradio.py` for the Gradio version
