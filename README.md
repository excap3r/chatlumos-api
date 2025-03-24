# PDF Wisdom Extractor

A tool for extracting concepts, questions, answers, and summaries from PDF lecture transcripts using LLMs and storing them in a MySQL database.

## Features

- Extract text from PDF files
- Process text with AI (DeepSeek, OpenAI, or Anthropic) to extract key concepts
- Store extracted data in MySQL database
- Generate concept visualizations
- Create summarized documents
- Track processing status and progress

## Requirements

- Python 3.7+
- MySQL server
- API key for at least one LLM provider (DeepSeek, OpenAI, or Anthropic)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pdf-wisdom-extractor.git
   cd pdf-wisdom-extractor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your database and API keys:
   
   Create a `.env` file in the project root with the following content:
   ```
   # LLM API Key (at least one is required)
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   # OPENAI_API_KEY=your_openai_key_here
   # ANTHROPIC_API_KEY=your_anthropic_key_here

   # MySQL configuration
   MYSQL_HOST=localhost
   MYSQL_USER=yourusername
   MYSQL_PASSWORD=yourpassword
   MYSQL_DATABASE=pdf_wisdom

   # Optional configuration
   # MAX_TOKENS_PER_CHUNK=4000    # Maximum tokens per text chunk
   # MODEL_NAME=deepseek-chat     # Model to use with DeepSeek
   # LLM_PROVIDER=deepseek        # Provider to use (deepseek, openai, anthropic)
   ```

4. Initialize the database:
   ```
   python pdf_wisdom_extractor.py --init_db
   ```

## Usage

### Process a PDF file

```
python pdf_wisdom_extractor.py --pdf_path path/to/your/document.pdf
```

Additional options:
- `--title "Document Title"` - Specify document title (default: filename)
- `--author "Author Name"` - Specify document author (default: "Iva Adamcová")
- `--chunk_size 1000` - Number of words per chunk
- `--chunk_overlap 200` - Overlap between chunks
- `--batch_size 5` - Process multiple chunks in parallel (requires multiple API keys)
- `--api_keys_file keys.txt` - File with multiple API keys (one per line)

### List all documents in the database

```
python pdf_wisdom_extractor.py --list_documents
```

### Create summary document for an existing document

```
python pdf_wisdom_extractor.py --document_id 1 --create_summary
```

### Create visualization for an existing document

```
python pdf_wisdom_extractor.py --document_id 1 --create_visualization
```

## Database Schema

The application uses the following database tables:

1. **documents** - Main document information
2. **document_chunks** - Individual text chunks from documents
3. **concepts** - Key concepts extracted from chunks
4. **qa_pairs** - Question-answer pairs extracted from chunks
5. **summaries** - Document summaries

## Output Files

When processing a document, the following files are generated:

- `concept_visualization.html` - Interactive visualization of key concepts
- `[filename]_summary.md` - Markdown summary of extracted concepts and Q&A pairs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 