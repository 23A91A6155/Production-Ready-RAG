# Vector Databases: A Deep Dive

## What Is a Vector Database?

A vector database is a specialized data store designed to efficiently index and query high-dimensional vectors. Unlike traditional relational databases that store structured rows and columns, vector databases store numerical arrays (vectors) and answer nearest-neighbor queries — finding which stored vectors are most similar to a given query vector.

Vector databases are the backbone of modern semantic search, recommendation systems, and RAG (Retrieval-Augmented Generation) pipelines.

## How Embeddings Are Generated

Embeddings are dense numerical representations of text (or images, audio, etc.) produced by neural networks. For text, an embedding model reads a piece of text and outputs a vector — a fixed-length list of floating-point numbers. OpenAI's text-embedding-3-small model, for example, produces 1536-dimensional vectors.

The key property of a good embedding model is that semantically similar texts produce vectors that are close together in the embedding space. This means that a question like "What is cosine similarity?" will produce a vector close to a chunk of text that explains cosine similarity, even if the two strings share few common words.

## Cosine Similarity vs. Euclidean Distance

Vector databases support different distance metrics. The two most common are:

**Cosine Similarity**: Measures the angle between two vectors. Ranges from -1 to 1; higher is more similar. Insensitive to vector magnitude, making it ideal for text embeddings where vector length can vary.

**Euclidean Distance (L2)**: Measures the straight-line distance between two points in vector space. Sensitive to magnitude. Can be misleading for high-dimensional text embeddings.

For RAG systems, cosine similarity is almost universally preferred because text embeddings tend to vary significantly in magnitude but carry their semantic meaning primarily in their direction.

## ChromaDB

ChromaDB is an open-source, embeddable vector database written in Python. Key features include:

- **Local Persistence**: Data is saved to disk using SQLite and a custom file format. No external server is required.
- **Cosine Similarity**: Supported as the default metric via the `hnsw:space` metadata parameter.
- **Collections**: Data is organized into named collections, each of which can have its own similarity metric.
- **Python-Native API**: Easy to use with `pip install chromadb`. No Docker or server setup needed.
- **Upsert Support**: ChromaDB supports idempotent upserts — inserting the same ID twice updates the existing record without duplication.

### ChromaDB Collection Configuration

When creating a ChromaDB collection for RAG, you typically configure:
- `name`: A unique identifier for the collection (e.g., "rag_chunks").
- `hnsw:space`: Set to "cosine" for cosine similarity search.

The collection stores documents (chunk text), embeddings (vectors), and metadata (source filename, chunk index) together, making retrieval self-contained.

## FAISS

FAISS (Facebook AI Similarity Search) is a library developed by Meta AI Research for efficient similarity search of dense vectors. It is optimized for:
- Very large-scale datasets (billions of vectors)
- GPU-accelerated search
- Multiple index types for different speed/accuracy tradeoffs

FAISS is typically used when maximum performance at scale is required. For local RAG projects with thousands to hundreds of thousands of chunks, ChromaDB is simpler to set up and sufficient for the workload.

## HNSW Index

Both ChromaDB and FAISS can use the HNSW (Hierarchical Navigable Small World) index structure. HNSW builds a multi-layer graph of vector neighbors, enabling approximate nearest-neighbor (ANN) search in sub-linear time. This makes querying fast even with millions of vectors, at the cost of small approximation error (sometimes a slightly non-optimal neighbor is returned).

For most RAG use cases, approximate nearest-neighbor search is perfectly acceptable because the small accuracy trade-off is imperceptible in practice.

## Persistence in RAG Systems

Persistence is critical in production RAG pipelines. Without persistence, embeddings would need to be regenerated every time the application starts — an expensive operation in both time and API cost. ChromaDB's `PersistentClient` stores all data to a local directory. Once ingestion completes, the vector store loads instantly on subsequent application starts, and queries can begin immediately.

## Metadata Storage

Beyond the embedding vectors, vector databases store metadata alongside each document. For RAG, this typically includes:
- **Source filename**: Which document this chunk came from.
- **Chunk index**: The position of this chunk within the source document.

This metadata is returned alongside query results, enabling the RAG system to display proper source citations to the user.

## Conclusion

Vector databases transform raw document embeddings into a queryable, persistent semantic index. Combined with embedding models and LLMs, they enable RAG systems to retrieve precise, relevant context from large document collections in milliseconds, grounding LLM answers in verifiable sources.
