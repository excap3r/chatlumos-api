# PDF Wisdom Extractor and Vector Database Tool

This project consists of three main components:

1. **PDF Wisdom Extractor** - Extracts concepts and Q&A pairs from PDF lecture transcripts
2. **MySQL to Vector Database Converter** - Converts extracted data to vector embeddings for AI-powered search
3. **Wisdom QA System** - Command-line interface for asking questions to the vector database

## Installation

### Prerequisites

- Python 3.8+
- MySQL database
- [Pinecone](https://www.pinecone.io/) account for vector database

### Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.template` to `.env` and fill in your API keys and database configuration:

```bash
cp .env.template .env
# Edit .env with your credentials
```

## PDF Wisdom Extractor Usage

The PDF Wisdom Extractor processes PDF lecture transcripts, extracts key concepts and Q&A pairs, and stores them in a MySQL database.

### Basic Usage

```bash
python pdf_wisdom_extractor.py --pdf_path path/to/lecture.pdf
```

### Options

- `--pdf_path`: Path to the PDF file to process
- `--chunk_size`: Number of words per chunk (default: 1000)
- `--chunk_overlap`: Number of overlapping words between chunks (default: 200)
- `--batch_size`: Number of chunks to process in parallel (default: 3)
- `--author`: Document author (default: "Iva Adamcová")
- `--title`: Document title (default: filename)
- `--translate_to_english`: Translate concepts and Q&A pairs to English
- `--list_documents`: List all documents in the database
- `--document_id ID`: Specify a document ID for operations
- `--export_data`: Export data to JSON files
- `--create_summary`: Create a summary document
- `--create_visualization`: Create a concept visualization

### Examples

Process a PDF with translation to English:
```bash
python pdf_wisdom_extractor.py --pdf_path lectures/wisdom.pdf --translate_to_english
```

List processed documents:
```bash
python pdf_wisdom_extractor.py --list_documents
```

Create a summary document:
```bash
python pdf_wisdom_extractor.py --document_id 1 --create_summary
```

## MySQL to Vector Database Converter

This tool converts concepts and Q&A pairs from the MySQL database to vector embeddings stored in Pinecone for efficient semantic search.

### Basic Usage

```bash
python mysql_to_vector.py
```

### Options

- `--document_id ID`: Process only a specific document
- `--model`: Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)
- `--index_name`: Name for the Pinecone index (default: wisdom-embeddings)
- `--list_documents`: List all documents in the database

### Examples

Convert all documents to vectors:
```bash
python mysql_to_vector.py
```

Process a specific document:
```bash
python mysql_to_vector.py --document_id 1
```

List documents:
```bash
python mysql_to_vector.py --list_documents
```

## Wisdom QA System

The Wisdom QA system provides a command-line interface for asking questions against your vector database. It uses DeepSeek to break down complex questions into sub-questions and concepts, then searches for relevant information in Pinecone.

### Basic Usage

Run the script with a question:
```bash
python wisdom_qa.py --question "What is the nature of reality according to Iva Adamcová?"
```

Or use interactive mode for continuous questioning:
```bash
python wisdom_qa.py --interactive
```

### Options

- `--question`: The question to ask
- `--model`: Embedding model to use (default: all-MiniLM-L6-v2)
- `--index`: Pinecone index name (default: wisdom-embeddings)
- `--top_k`: Number of results to retrieve per query (default: 10)
- `--interactive`: Run in interactive mode

### Examples

Ask a specific question:
```bash
python wisdom_qa.py --question "What is the law of attraction?"
```

Use a specific index and more results:
```bash
python wisdom_qa.py --question "How can I achieve inner peace?" --index my-custom-index --top_k 20
```

Run in interactive mode:
```bash
python wisdom_qa.py --interactive
```

## Complete Workflow

For a complete extraction-to-search workflow:

1. Extract wisdom from PDF:
```bash
python pdf_wisdom_extractor.py --pdf_path lectures/wisdom.pdf --translate_to_english
```

2. Convert to vector database:
```bash
python mysql_to_vector.py
```

3. Ask questions using the QA system:
```bash
python wisdom_qa.py --interactive
```

## Required Packages

Create a `requirements.txt` file with:

```
PyPDF2>=3.0.0
mysql-connector-python>=8.0.0
python-dotenv>=0.20.0
requests>=2.27.1
tqdm>=4.64.0
tiktoken>=0.4.0
pinecone-client>=2.2.1
torch>=1.13.0
sentence-transformers>=2.2.2
numpy>=1.23.0
```

## License

MIT 