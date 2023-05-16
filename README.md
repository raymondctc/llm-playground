# LLM Playground

This is a LLM playground

## Setup

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/mac-os/)
2. Create a virtual environment for Python

   ```bash
   conda create -n myenv python=3.10
   ```

3. Run `pip install -r requirements.txt`
4. Register an account on OpenAI, HuggingFace and Pinecone. Get the API keys/access tokens.
5. Put the keys/access tokens in `.env` file. (Check `.env.template`)

## Apps

`analyer.py` - a showcase of analyzing Play Store Review with LangChain's retrieval API, HuggingFace embeddings and  Pinecone vector store.

- To use Pinecone, you have to create an index on Pinecone. (Use 768 as dimension size if you're using `HuggingFaceEmbeddings`, use 1536 if you're using `OpenAIEmbeddings`)
