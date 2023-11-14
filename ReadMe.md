# 🦙 YouTube In-Video Search Engine

This Streamlit-based application is a powerful tool for searching and analyzing transcripts from YouTube videos. By integrating advanced NLP techniques and leveraging the Llama-2 model from Hugging Face, along with Torch, Langchain, and the YouTube Transcript API, it offers a comprehensive approach to processing and querying video transcripts. Additionally, the application utilizes VectorDB for storing and indexing transcripts and implements simple Retrieval Augmented Generation (RAG) pipeline for enhanced query understanding and response generation.

## Features

- **Transcript Extraction**: Extract transcripts from YouTube videos using the YouTube Transcript API.
- **Transcript Indexing**: Utilize VectorDB to store and index transcripts for efficient querying.
- **Advanced Query Processing**: Implement the Retrieval Augmented Generation (RAG) pipeline for improved natural language understanding and response generation.
- **Llama-2 Model Integration**: Leverage the Llama-2 model from Hugging Face for robust NLP capabilities.
- **Interactive User Interface**: Streamlit-based UI for easy input of YouTube URLs and search queries.


## Installation

To set up this project, clone the repository and install the required dependencies.

### Clone the Repository

```bash
git clone https://github.com/farooquiowais/Llama2-In-Video-Search-Engine.git
cd Llama2-In-Video-Search-Engine
```

### Install Dependencies

Ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:
```
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
langchain einops accelerate transformers bitsandbytes scipy
xformers sentencepiece
llama-index==0.7.21 llama_hub==0.0.19
pydantic==1.10.9
sentence-transformers
openai==0.28.0
youtube_transcript_api
streamlit
```

## Usage

To start the application, run:

```bash
streamlit run app.py
```

Navigate to the provided localhost URL in your web browser to interact with the application.


## License

Distributed under the MIT License. See `LICENSE` for more information.