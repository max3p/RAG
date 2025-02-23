
# RAG with Ollama and ChromaDB

This project demonstrates a local Retrieval Augmented Generation (RAG) pipeline that processes PDFs, builds a vector database from document chunks, and answers queries using a local language model (LLM) served by Ollama. The system leverages LangChain for document processing, ChromaDB for vector storage, and either AWS Bedrock or Ollama for embeddings.

---

## Tech Stack

- **Python**
- **LangChain**: For PDF parsing and text splitting  
  - [LangChain Docs](https://python.langchain.com/docs/introduction/)
  - [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- **Embedding Providers:**
  - **AWS Bedrock** (default embedding function provided by `langchain-community`)
  - **Ollama Embedding** (optional; note that local embeddings may not be as good)
- **ChromaDB**: Vector database stored on the filesystem
- **PyPDF**: PDF loader for extracting document content
- **Ollama**: LLM Processor
- **FastAPI**: Python API (planned – TODO)
- **Docker**: Container deployment (planned – TODO)

---

## Local Ollama Setup

To run Ollama locally, follow these steps:

1. **Download Ollama:**  
   Visit [Ollama](https://ollama.com/) and install the application.

2. **Install Desired Model:**  
   For example, to install the *mistral* model, run:
   ```bash
   ollama pull mistral
   ```
   *(You can also pull other models like llama2, deepseek, etc.)*

3. **Serve the API:**  
   Start the local Ollama API server by running:
   ```bash
   ollama serve
   ```
   Access it via: `http://localhost:11434`

**Note:**  
While you can generate embeddings locally with Ollama, it’s recommended to use AWS Bedrock or a similarly powerful service since local embeddings might not perform as well. If you have a beefy computer and a larger open-source model, you may consider switching to local embeddings.

To switch the embedding function to use Ollama locally, update `get_embedding_function.py` by commenting out the Bedrock code and uncommenting the Ollama code:
```python
# embeddings = BedrockEmbeddings(...)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## Setup & Installation

1. **Clone the Repository**  
   Ensure all files are located within a directory named `RAG`. The structure should look like:
   ```
   RAG/
   ├── data/
   │   └── yourfile.pdf
   ├── get_embedding_function.py
   ├── populate_database.py
   ├── query_data.py
   └── test_rag.py
   ```

2. **Install Dependencies**  
   Make sure you have all required Python packages installed. If you have a `requirements.txt`, run:
   ```bash
   pip install -r requirements.txt
   ```
   Otherwise, install the necessary packages manually.

3. **Set Up Ollama & AWS Credentials** (if using AWS Bedrock)  
   - For AWS Bedrock embeddings, ensure you have your AWS credentials configured (e.g., via `aws configure` or your environment).
   - For local embedding with Ollama, update `get_embedding_function.py` as mentioned above.

---

## Running the Project

### 1. Populate the Database

Before querying, populate the Chroma vector database with your PDF data:

- **Run the Script:**
  ```bash
  python populate_database.py
  ```
- **Optional – Reset Database:**  
  To clear any existing data and start fresh, run:
  ```bash
  python populate_database.py --reset
  ```

### 2. Query the Database

After populating the database, you can ask questions to the system:

- **Run the Query Script:**  
  Replace `"Your question here"` with your actual query.
  ```bash
  python query_data.py "Your question here"
  ```

  The script will:
  - Load the vector database,
  - Perform a similarity search for the most relevant document chunks,
  - Construct a prompt using the retrieved context and your query,
  - Send the prompt to your local LLM (Ollama using the `mistral` model),
  - Output the answer along with source identifiers.

### 3. Running Tests

You can run tests to validate the response accuracy using the provided test harness:

- **Run the Test Script:**
  ```bash
  python test_rag.py
  ```

  This script contains hardcoded test cases (e.g., basic arithmetic and color mixing questions) that compare the model's responses to expected outputs.

---

## Future Work (TODO)

- **FastAPI Integration:**  
  Develop a FastAPI application to expose the RAG functionality as a web API.
- **Docker Deployment:**  
  Containerize the application using Docker for easier deployment and scalability.

---

## Additional Resources

- **LangChain Documentation:**  
  [https://python.langchain.com/docs/introduction/](https://python.langchain.com/docs/introduction/)
- **Ollama:**  
  [https://ollama.com/](https://ollama.com/)

---