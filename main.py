
import argparse
import sys
from pathlib import Path

# LangChain imports - try modern modular imports first, fallback to legacy if needed
try:
    # Modern LangChain structure (v0.1.0+)
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
except ImportError:
    try: 

        # Legacy LangChain structure (pre-0.1.0)
        from langchain.document_loaders import TextLoader  # type: ignore
        from langchain.text_splitters import CharacterTextSplitter  # type: ignore
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain.vectorstores import Chroma  # type: ignore
        from langchain.llms import Ollama  # type: ignore
    except ImportError as e:
        print(f"ERROR: Failed to import required LangChain modules. {e}")
        print("Please ensure langchain and langchain-community are installed:")
        print("  pip install langchain langchain-community langchain-text-splitters")
        sys.exit(1)


SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def build_vector_store(speech_path: str = SPEECH_FILE, persist_dir: str = CHROMA_DIR):
    """
    Load speech text, split into chunks, create embeddings, and persist a Chroma vector store.
    """
    if not Path(speech_path).is_file():
        print(f"ERROR: {speech_path} not found.")
        sys.exit(1)

    print("Loading speech.txt ...")
    loader = TextLoader(speech_path, encoding="utf-8")
    documents = loader.load()

    print("Splitting text into chunks ...")
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    docs = splitter.split_documents(documents)

    print("Creating embeddings ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building Chroma vector store ...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # Chroma 0.4.x automatically persists, so manual persist() is no longer needed
    # vectordb.persist()  # Deprecated in Chroma 0.4.x
    print("Vector store created.")
    return vectordb


def load_vector_store(persist_dir: str = CHROMA_DIR):
    """
    Load an existing Chroma vector store from disk.
    """
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"Chroma directory '{persist_dir}' not found.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb


def create_qa_chain(vectordb):
    """
    Create a simple QA chain that retrieves relevant chunks from the vector store
    and queries a local Ollama model (mistral). This avoids requiring LangChain's
    high-level `RetrievalQA` if the installed LangChain version doesn't match.
    """
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    class SimpleQA:
        def __init__(self, retriever, model_name: str = "mistral"):
            self.retriever = retriever
            self.model_name = model_name

        def __call__(self, inputs):
            query = inputs.get("query") if isinstance(inputs, dict) else str(inputs)
            # Retrieve relevant documents using LangChain retriever API
            # Try different methods based on LangChain version
            try:
                # Standard method: use get_relevant_documents (works in most LangChain versions)
                if hasattr(self.retriever, "get_relevant_documents"):
                    docs = self.retriever.get_relevant_documents(query)
                # Modern LangChain: try invoke method (LCEL interface)
                elif hasattr(self.retriever, "invoke"):
                    docs = self.retriever.invoke(query)
                    # invoke may return a list directly or wrapped
                    if not isinstance(docs, list):
                        docs = list(docs) if docs else []
                else:
                    raise AttributeError(
                        "Retriever does not have 'get_relevant_documents' or 'invoke' method. "
                        f"Retriever type: {type(self.retriever)}, "
                        f"Available methods: {[m for m in dir(self.retriever) if not m.startswith('_')]}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Error retrieving documents: {e}\n"
                    "Please ensure the vector store is properly initialized."
                ) from e

            context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

            prompt = (
                "Use only the following context from the Ambedkar speech to answer the question."
                " If the answer is not present in the context, reply: 'I don't know based on the provided text.'\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )

            # Use LangChain's Ollama wrapper (already imported)
            answer = None
            try:
                llm = Ollama(model=self.model_name)
                # LangChain LLMs should use invoke() method (or predict() for older versions)
                if hasattr(llm, "invoke"):
                    answer = llm.invoke(prompt)
                elif hasattr(llm, "predict"):
                    answer = llm.predict(prompt)
                elif hasattr(llm, "__call__"):
                    answer = llm(prompt)
                else:
                    raise AttributeError("LLM does not have invoke, predict, or __call__ method")
                # Extract text if answer is not a string
                if isinstance(answer, dict):
                    answer = answer.get("text", answer.get("content", str(answer)))
                elif not isinstance(answer, str):
                    answer = str(answer)
            except Exception as e:
                # Fallback: try direct ollama CLI if LangChain wrapper fails
                try:
                    import subprocess
                    import shutil
                    # Try to find ollama executable in PATH
                    ollama_cmd = shutil.which("ollama") or "ollama"
                    completed = subprocess.run(
                        [ollama_cmd, "run", self.model_name, prompt],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                        shell=False  # shell=False for Windows compatibility
                    )
                    answer = completed.stdout.strip()
                except Exception as cli_error:
                    answer = (
                        f"Error: Unable to use Ollama LLM. "
                        f"Please ensure:\n"
                        f"  1. Ollama is installed (https://ollama.ai/)\n"
                        f"  2. Mistral model is pulled: 'ollama pull {self.model_name}'\n"
                        f"  3. Ollama service is running\n"
                        f"LangChain error: {e}\n"
                        f"CLI error: {cli_error}"
                    )

            return {"result": str(answer), "source_documents": docs}

    return SimpleQA(retriever)


def cli_loop(qa_chain):
    """
    Start the command-line question loop.
    """
    print("\nAmbedkarGPT - Q&A System (based only on speech.txt)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not query:
            continue

        response = qa_chain({"query": query})
        answer = response.get("result", "")
        sources = response.get("source_documents", [])

        print("\nAnswer:\n")
        print(answer.strip())
        print("\nRetrieved Context (showing up to 3 chunks):")
        for i, doc in enumerate(sources[:3], start=1):
            print(f"\n--- chunk {i} ---")
            print(doc.page_content.strip())

        print("\n" + ("-" * 50) + "\n")


def check_ollama_connection():
    """
    Check if Ollama service is running and accessible.
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if "mistral" in str(model_names).lower():
                return True, "Ollama is running and Mistral model is available."
            else:
                return False, "Ollama is running but Mistral model not found. Run: ollama pull mistral"
        return False, "Ollama service responded but with error."
    except requests.exceptions.ConnectionError:
        return False, "Ollama service is not running. Please start Ollama first."
    except ImportError:
        # If requests not available, try simple check with Ollama CLI
        try:
            import subprocess
            import shutil
            ollama_cmd = shutil.which("ollama") or "ollama"
            result = subprocess.run(
                [ollama_cmd, "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "mistral" in result.stdout.lower():
                return True, "Ollama CLI accessible and Mistral model found."
            elif result.returncode == 0:
                return False, "Ollama CLI accessible but Mistral model not found. Run: ollama pull mistral"
            else:
                return False, "Ollama CLI check failed."
        except Exception:
            return None, "Cannot verify Ollama. Please ensure Ollama is installed and running."


def main():
    parser = argparse.ArgumentParser(description="AmbedkarGPT - Command-line RAG Q&A System")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store and embeddings")
    parser.add_argument("--skip-ollama-check", action="store_true", help="Skip Ollama connection check")
    args = parser.parse_args()

    if not Path(SPEECH_FILE).is_file():
        print(f"ERROR: '{SPEECH_FILE}' is missing.")
        sys.exit(1)

    # Check Ollama connection before proceeding
    if not args.skip_ollama_check:
        print("Checking Ollama connection...")
        status, message = check_ollama_connection()
        if status is False:
            print(f"⚠️  WARNING: {message}")
            print("\nTo start Ollama:")
            print("  1. Windows: Start Ollama from Start Menu or run 'ollama serve' in a terminal")
            print("  2. Mac/Linux: Run 'ollama serve' or ensure Ollama service is running")
            print("  3. Pull Mistral model: 'ollama pull mistral'")
            print("\nContinuing anyway... (use --skip-ollama-check to suppress this warning)\n")
        elif status is True:
            print(f"✅ {message}\n")

    if args.rebuild:
        print("Rebuilding vector store.")
        vectordb = build_vector_store()
    else:
        try:
            vectordb = load_vector_store()
            print("Loaded existing vector store.")
        except Exception:
            print("No existing vector store found. Creating new one.")
            vectordb = build_vector_store()

    qa_chain = create_qa_chain(vectordb)
    cli_loop(qa_chain)


if __name__ == "__main__":
    main()
