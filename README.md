# Vacancy Skill Analyzer

**Vacancy Skill Analyzer** is a tool that extracts and analyzes job posting content to identify and quantify in-demand skills. It uses LLMs (Mistral 7B via `llama-cpp`) to parse unstructured job descriptions and classify skills into predefined categories, producing a frequency distribution of relevant technologies and qualifications.

## Features

* 🔗 Fetches and cleans job postings from URLs
* 🧠 Uses LLM to extract concise skill requirements from vacancy text
* 🧹 Filters and classifies extracted skills into defined categories
* 📊 Aggregates and visualizes skill frequencies

## Setup

### Requirements

* Python 3.10
* llama-cpp-python
* pandas
* tqdm
* beautifulsoup4
* requests

### Installation

```bash
pip install llama-cpp-python pandas tqdm beautifulsoup4 requests
```

### Download Mistral Model

Place the Mistral 7B instruct model (`*.gguf`) in the `models/` directory. You can use `mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

## Usage

1. Add URLs of job postings to the `urls` list in `vacancies.py`
2. Run the script:

```bash
python vacancies.py
```

3. Results will be printed as a sorted Dictionary.
