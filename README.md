# LLM your Docs Localy
### Use a local Large Language Model to read and provide answers on your local files

<hr>

Uses https://huggingface.co/mosaicml/mpt-7b-instruct, a model published by
<br>
Mosaic ML https://mosaicml.com 


PDF documents in the `docs` folder are loaded into a vectorstore
<br>
Includes EDPannualreport.pdf as example. 
<br>
It is the Annual Report from EDP, a large company from the energy sector

The model can then answer questions about the ducument(s) it has ingested.

There are two versions one CLI, and one GUI based on Gradio

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

Note: Not tested, do not use for production
