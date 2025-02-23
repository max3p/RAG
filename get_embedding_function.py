#get_embedding_function.py
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# Text embedding function from AWS Bedrock
# Text use Ollama (locally hosted) but its not as good
def get_embedding_function():
    #embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
