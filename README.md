# AmbedkarGPT-Intern-Task

A command-line Q&A system that answers questions based solely on a provided speech excerpt from Dr. B.R. Ambedkar's "Annihilation of Caste". The system uses a RAG (Retrieval-Augmented Generation) pipeline to retrieve relevant context and generate accurate answers.

## Architecture

The system implements a complete RAG pipeline:

1. **Document Loading**: Loads text from `speech.txt` using LangChain's `TextLoader`
2. **Text Chunking**: Splits the text into manageable chunks (400 chars with 80 char overlap) using `CharacterTextSplitter`
3. **Embedding Generation**: Creates vector embeddings using `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model
4. **Vector Storage**: Stores embeddings in ChromaDB (local, open-source vector database)
5. **Retrieval**: Retrieves relevant chunks based on semantic similarity to user questions
6. **Answer Generation**: Uses Ollama with Mistral 7B LLM to generate answers from retrieved context

## Project Files

- `main.py` - Main application code implementing the RAG pipeline
- `speech.txt` - Text excerpt from Dr. B.R. Ambedkar's speech
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running (see setup instructions below)

## Setup Instructions

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama

Ollama is a separate application (not a Python package) that runs LLMs locally.

**Windows:**
1. Download Ollama installer from:` https://ollama.ai/download`
2. Run the installer (usually `ollama-windows-amd64.exe` or similar)
3. The installer will add Ollama to your PATH automatically
4. **IMPORTANT**: After installation, you may need to:
   - Close and reopen your terminal/PowerShell
   - Or restart your computer for PATH changes to take effect
5. Verify installation by opening a **new** terminal and running:
   ```bash
   ollama --version
   ```
   If you see a version number, Ollama is installed correctly.

**Mac/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 4: Start Ollama Service and Pull Mistral Model

**Windows:**
Ollama typically runs as a background service on Windows. To start it:
1. Open a **new** terminal/PowerShell window
2. Run: `ollama serve`
   - Keep this terminal open while using the Q&A system
   - You should see "Ollama is running" message

**Mac/Linux:**
```bash
ollama serve
```

Then, in another terminal window, pull the Mistral 7B model:

```bash
ollama pull mistral
```

This will download the model (approximately 4.1GB). Wait for the download to complete.

### Step 5: Verify speech.txt

Ensure `speech.txt` exists in the project root and contains the provided text excerpt.

## Running the Program

### First Run

On the first run, the system will create the vector store:

```bash
python main.py
```

The system will:
1. Load `speech.txt`
2. Split it into chunks
3. Generate embeddings
4. Store them in ChromaDB
5. Start the Q&A interface

### Subsequent Runs

After the first run, the vector store is saved locally in the `chroma_store/` directory. The system will automatically load it on subsequent runs.

### Force Rebuild

To rebuild the vector store from scratch (e.g., if you modify `speech.txt`):

```bash
python main.py --rebuild
```

### Using the Q&A Interface

Once the system is running:
- Type your question and press Enter
- The system will retrieve relevant chunks and generate an answer
- Type `exit` or `quit` to stop the program
- Press `Ctrl+C` to exit at any time

### Example Questions

- "What is the real remedy according to Ambedkar?"
- "What does Ambedkar say about the shastras?"
- "What is the problem of caste according to this text?"
- "What is the analogy about the gardener?"

## Technical Stack

- **Framework**: LangChain (orchestration and RAG pipeline)
- **Vector Database**: ChromaDB (local, open-source)
- **Embeddings**: HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM**: Ollama with Mistral 7B model
- **No API Keys Required**: Everything runs locally and offline

## Project Structure

```
AmbedkarGPT/
├── main.py              # Main application code
├── speech.txt           # Source text file
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── chroma_store/       # Generated vector store (created on first run)
```

## Troubleshooting

**Issue: "ERROR: speech.txt not found"**
- Ensure `speech.txt` is in the project root directory

**Issue: "ollama: command not found" or "Ollama is not recognized"**
- Ollama is not installed or not in PATH
- Download and install from: https://ollama.ai/download
- After installation, **close and reopen** your terminal
- Verify installation: `ollama --version`

**Issue: "Ollama LLM unavailable" or "Connection refused"**
- Verify Ollama is installed: `ollama --version`
- **Start Ollama service**: Open a new terminal and run `ollama serve`
  - Keep that terminal open while using the system
  - You should see "Ollama is running" message
- Ensure Mistral model is pulled: `ollama pull mistral`
- Test Ollama directly: `ollama run mistral "Hello"`

**Issue: Import errors**
- Make sure you're in the virtual environment
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

**Issue: ChromaDB errors**
- Delete the `chroma_store/` directory and rebuild: `python main.py --rebuild`

## Notes

- The system only answers questions based on the provided text in `speech.txt`
- If the answer is not in the context, the system will indicate that
- All processing happens locally - no data is sent to external services
- The first run may take longer as it downloads the embedding model and creates the vector store
