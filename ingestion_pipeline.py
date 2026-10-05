import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import re

# extract text from file
with open("company_policy.txt", "r", encoding="utf-8") as f:
    text = f.read()

#data parsing(chunking)
chunks= [chunk.strip() for chunk in re.split(r"[,\.]",text) if chunk.strip()]

#print(f"loaded {len(chunks)} {chunks} from the file")

#initilizing Chromadb
client= chromadb.PersistentClient(path= "./chromadb")

#connect to Ollama's embedding function to convert data into vectors
ef= OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

collection= client.get_or_create_collection(
    name= "company_policy",
    embedding_function= ef,
)

#add chunks to the collection-Chromadb automatically generates embeddings 
collection.add(ids=[f"chunks{i}" for i in range(len(chunks))],
               documents=chunks,
               metadatas=[{"source":"company_policy.txt", "chunk_index": i} for i in range(len(chunks))]
               #metadata actually tells us about the chunks description
               )
print(f"added {len(chunks) } chunks to collection")

