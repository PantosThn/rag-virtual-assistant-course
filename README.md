# RAG Virtual Assistant Course

A repository for building a Retrieval-Augmented Generation (RAG) virtual assistant.

## 1. Clone the repository

```bash
git clone https://github.com/PantosThn/rag-virtual-assistant-course.git
cd rag-virtual-assistant-course
```

## 2. Install Python 3.10

Ensure you have **Python 3.10** installed.

- **macOS (Homebrew)**  
  ```bash
  brew install python@3.10
  ```
- **Ubuntu / Debian**  
  ```bash
  sudo apt update
  sudo apt install python3.10 python3.10-venv python3.10-dev
  ```
- **Windows**  
  Download and install from the [official Python 3.10 release](https://www.python.org/downloads/release/python-3100/).

Verify the installation:

```bash
python3.10 --version
```

> If `python3.10` isn’t recognized on Windows, try `python --version`.

## 3. Install Poetry

### macOS & Linux

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Windows (Git Bash)

1. **Add Poetry to your PATH**  
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```
2. **Install pipx & Poetry**  
   ```bash
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install poetry
   ```
3. **Verify**  
   ```bash
   poetry --version
   # → Poetry (version X.Y.Z)
   ```

### Windows (PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
poetry --version
```

## 4. Install project dependencies

Inside the project root:

```bash
poetry install
```

## 5. Usage

- **Activate a shell**  
  ```bash
  poetry shell
  ```
- **Run a script**  
  ```bash
  poetry run python your_script.py
  ```

---

Feel free to customize any step to fit your environment!