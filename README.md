# Semantic Spotter: Fashion Product Search System

## Overview

Semantic Spotter is an intelligent fashion product search system that leverages Retrieval-Augmented Generation (RAG) to provide contextual product recommendations from the Myntra fashion dataset. The system combines semantic search capabilities with large language models to deliver accurate and relevant fashion product suggestions.

## Features

- **Semantic Search**: Uses HuggingFace embeddings for understanding product queries
- **Cross-Platform Support**: Works on Kaggle, Google Colab, and local environments
- **Document Reranking**: Implements cross-encoder reranking for improved search relevance
- **Multi-format Support**: Handles PDF, CSV, and text file formats
- **Metadata Enhancement**: Extracts and enriches product metadata including images, prices, ratings
- **Conversational AI**: Powered by Perplexity's Sonar model for natural language responses

## Architecture

```
User Query → Vector Search → Cross-Encoder Reranking → LLM Response Generation → Final Answer
```

### Core Components

1. **Document Processing Pipeline**
   - Multi-format document loaders (PDF, CSV, TXT)
   - Recursive text splitting with configurable chunk sizes
   - Metadata extraction and enhancement

2. **Vector Store**
   - FAISS vector database for efficient similarity search
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Persistent storage with refresh capabilities

3. **Retrieval System**
   - Contextual compression retriever
   - Cross-encoder reranking (BAAI/bge-reranker-base)
   - Top-k document retrieval

4. **Language Model Integration**
   - Perplexity Sonar-Pro model
   - Fashion domain-specific prompting
   - Context-aware response generation

## Installation

### Required Dependencies

```bash
pip install -qU langchain-community
pip install -qU langchain_huggingface
pip install -qU sentence-transformers
pip install -qU langchain-perplexity
pip install -qU faiss-cpu
pip install -qU gradio
pip install -qU kaggle
```

### Environment Setup

The system automatically detects the platform and configures accordingly:

- **Kaggle**: Uses Kaggle secrets for API keys
- **Google Colab**: Uses Colab userdata for API keys
- **Local**: Uses .env file for environment variables

Required environment variables:
- `PERPLEXITY_API_KEY` or `PPLX_API_KEY`: Perplexity API key for LLM access

## Configuration

The system uses a centralized configuration dictionary:

```python
config = {
    'data_path': '/path/to/dataset/',
    'images_path': '/path/to/images/',
    'chunk_size': 512,
    'chunk_overlap': 80,
    'vector_store_name': "faiss_myntra_db",
    'embedding_model': 'all-MiniLM-L6-v2',
    'refresh_vector_store': 'N',
    'domain': 'fashion',
    'chat_model': "sonar-pro",
    'rerank_model': 'BAAI/bge-reranker-base'
}
```

### Key Parameters

- **chunk_size**: Size of text chunks for processing (default: 512)
- **chunk_overlap**: Overlap between consecutive chunks (default: 80)
- **refresh_vector_store**: Whether to rebuild the vector store ('Y'/'N')
- **embedding_model**: HuggingFace model for text embeddings
- **rerank_model**: Cross-encoder model for result reranking

## Dataset

The system uses the Myntra Fashion Product Dataset containing:
- Product descriptions and metadata
- Product images
- Pricing information
- Brand details
- Customer ratings and reviews

### Supported Data Formats
- **PDF**: Product catalogs and documentation
- **CSV**: Structured product data
- **TXT**: Plain text product descriptions

## Core Functions

### Document Processing

#### `add_metadata_to_documents(documents)`
Enhances document metadata by extracting:
- Product ID and name
- Category and brand information
- Pricing details
- Image URLs and local paths
- Rating information

#### `get_data_chunks(folder_path)`
Processes documents from a directory:
- Automatically detects file types
- Applies appropriate loaders
- Creates text chunks with metadata
- Returns processed document chunks

### Vector Store Management

#### `get_embeddings_model()`
Initializes HuggingFace embeddings model with:
- GPU acceleration support
- Multi-processing capabilities
- Progress tracking

#### `create_vector_store(text_chunks, embedding_model)`
Creates or loads FAISS vector store:
- Conditional refresh based on configuration
- Persistent storage for efficiency
- Dangerous deserialization handling

### Retrieval System

#### `get_retriever(top_k=10)`
Creates basic vector store retriever with configurable top-k results.

#### `get_reranked_query_results(query)`
Implements advanced retrieval with:
- Cross-encoder reranking
- Contextual compression
- Top-3 result selection

### Response Generation

#### `generate_llm_response(query, results)`
Generates contextual responses using:
- Fashion domain-specific system prompts
- Perplexity Sonar model
- Metadata extraction instructions

#### `rag_pipeline(user_input)`
Complete RAG pipeline:
1. Retrieves relevant documents
2. Formats context with metadata
3. Generates LLM response
4. Returns final answer

## Usage Examples

### Basic Query
```python
user_input = "party dresses for women"
response = get_answer(user_input)
print(response.content)
```

### Custom Configuration
```python
config['chunk_size'] = 256
config['refresh_vector_store'] = 'Y'
# Rebuild vector store with new settings
```

## Platform-Specific Features

### Kaggle Integration
- Automatic dataset access
- Kaggle secrets management
- GPU acceleration support

### Google Colab Support
- Colab userdata integration
- Drive mounting capabilities
- T4 GPU optimization

### Local Development
- Environment variable loading
- Custom dataset paths
- Development debugging features

## Performance Optimization

### Vector Store Efficiency
- Persistent FAISS storage
- Conditional refresh mechanism
- GPU-accelerated embeddings

### Retrieval Optimization
- Cross-encoder reranking
- Contextual compression
- Configurable result limits

### Memory Management
- Chunk-based processing
- Lazy loading mechanisms
- Efficient metadata handling

## Error Handling

The system includes robust error handling for:
- Missing metadata fields
- File format incompatibilities
- API key configuration issues
- Platform detection failures

## Limitations

- Requires Perplexity API access
- Limited to fashion domain products
- Dependent on dataset quality
- GPU recommended for optimal performance

## Future Enhancements

- Multi-modal search with image inputs
- Real-time product availability
- User preference learning
- Advanced filtering capabilities
- Gradio web interface integration

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement changes with proper documentation
4. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Support

For issues and questions:
- Check the documentation
- Review configuration settings
- Verify API key setup
- Ensure dataset accessibility

---

*Semantic Spotter - Intelligent Fashion Discovery through AI*
