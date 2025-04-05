import faiss

# Path to your local FAISS index file
index = faiss.read_index("/Users/pranavkulkarni/Downloads/Test_Course_faiss.index")

# Check how many vectors and their dimension
print("Number of vectors indexed:", index.ntotal)
print("Vector dimension:", index.d)

# Optionally: perform a test search
import numpy as np
dummy_query = np.random.random((1, index.d)).astype('float32')
distances, indices = index.search(dummy_query, k=3)
print("Test search result:")
print("Indices:", indices)
print("Distances:", distances)
