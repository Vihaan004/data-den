# ChromaDB Vector Store Implementation

This document explains how ChromaDB has been integrated into the GPU Mentor application as a persistent vector store.

## Overview

The application now supports two types of vector stores:
1. **In-Memory Vector Store** (original implementation): Fast but temporary
2. **ChromaDB Vector Store** (new implementation): Persistent storage that preserves embeddings between sessions

## Configuration

The vector store behavior can be controlled through settings in `config.py`:

```python
# Vector Store Configuration
USE_PERSISTENT_VECTORSTORE = True  # Set to False to use in-memory
VECTORSTORE_PERSIST_DIRECTORY = "./output/vectorstore_data"  # Directory for persistent vector storage
```

## Benefits of ChromaDB

1. **Persistence**: Vector embeddings are stored on disk and preserved between application restarts
2. **Faster Startup**: No need to re-embed documents on each startup (significant time savings)
3. **Incremental Updates**: New documents can be added without rebuilding the entire database
4. **Scalability**: Can handle larger document collections than in-memory solutions

## How It Works

The `VectorStore` class in `document_loader.py` now:
1. Checks if persistent storage is enabled in the config
2. Checks if a ChromaDB instance already exists on disk
3. Loads the existing store or creates a new one
4. Exposes the same retriever interface so other components aren't affected

## Usage

The implementation is designed to be a drop-in replacement for the original in-memory vector store. No changes are needed to other components.

### Trying It Out

Run the `chroma_demo.py` script to test the ChromaDB implementation:

```bash
python chroma_demo.py
```

The first run will create a new ChromaDB store. Subsequent runs will load the existing store.

### Switching Between Implementations

To switch between in-memory and persistent storage, simply change the `USE_PERSISTENT_VECTORSTORE` setting in `config.py`.

### Clearing the Database

If you need to rebuild the vector store from scratch, delete the directory specified in `VECTORSTORE_PERSIST_DIRECTORY`.

## Advanced Usage

The `VectorStore` class now includes these additional methods:

- **add_documents()**: Add new documents to an existing vector store
- **get_document_count()**: Get the number of documents in the store
- **_check_existing_vectorstore()**: Check if a persistent store exists
- **load_vectorstore()**: Load an existing persistent store

## Troubleshooting

- If you encounter errors, ensure ChromaDB is installed: `pip install chromadb>=0.4.18`
- If the vector store isn't loading, check if the directory exists and has content
- For permission errors, ensure the application has write access to the directory
