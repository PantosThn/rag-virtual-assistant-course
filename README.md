
# RAG Virtual Assistant Course

Welcome! This is a 4-part hands-on tutorial series where you will build a virtual assistant using Retrieval-Augmented Generation (RAG) techniques and modern LLM tools.

Each lesson is inside `notebooks/`, progressively guiding you through:

- Retrieval systems
- Vector databases
- Prompt chaining
- Building a full RAG-powered assistant

## 🔑 API Keys Setup

Before running the lessons, make sure you have the following API keys:

- **OpenAI API Key**: Create or view your key here → [OpenAI API Keys](https://platform.openai.com/api-keys)
- **LangSmith API Key**: Create or manage your key here → [LangSmith Settings](https://smith.langchain.com/o/0a40799b-2bb0-55cc-a7b6-c65896228e62/settings)


## Quick Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-virtual-assistant-course.git
cd rag-virtual-assistant-course
```

### 2. Install Python 3.10 (if not already installed)

Ensure you're using Python 3.10.

- On MacOS with Homebrew:
  ```bash
  brew install python@3.10
  ```

- On Ubuntu/Debian:
  ```bash
  sudo apt update
  sudo apt install python3.10 python3.10-venv python3.10-dev
  ```

- On Windows:
  Download Python 3.10 from the [official website](https://www.python.org/downloads/release/python-3100/).

Verify version:

```bash
python3.10 --version
```

---

### 3. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

---

### 4. Install project dependencies

```bash
poetry env use python3.10
poetry install
```

---

### 5. Create your `.env` file

Create a `.env` file in the project root and paste:

```env
OPENAI_API_KEY=your-openai-key
LANGCHAIN_API_KEY=your-langchain-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
USER_AGENT=Mozilla/5.0 (compatible; RAG-TutorialBot/1.0; +https://yourwebsite.com/bot)
```

> 🔒 **Important**: Never commit your `.env` file.

---

### 6. Activate the environment

```bash
poetry shell
```

---

### 7. Launch the notebooks

```bash
poetry run jupyter notebook
```
Then open the `notebooks/` folder and start with `lesson1_intro_to_rag.ipynb`.

---

## Requirements

- Python 3.10
- Poetry
- OpenAI API key
- (Optional) LangSmith account for tracing

---

## Notes

- This repo uses **Poetry** for dependency management.
- Environment variables are loaded securely using **python-dotenv**.
- Designed for clean local development and easy reproducibility.

---

Happy learning! 🚀
