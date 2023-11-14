# Streamlit Application Development Imports
import streamlit as st
import re
from pathlib import Path

# Transformers and PyTorch Imports for Language Model
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# Llama Index and Embedding Imports for Document Processing
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext
from llama_index import VectorStoreIndex
from llama_index.schema import Document

# YouTube Transcript API Import
from youtube_transcript_api import YouTubeTranscriptApi

#Importing helper functions
from helper.functions import get_url_id, get_tokenizer_model

# Create a tokenizer and model
tokenizer, model = get_tokenizer_model()

# Create a new instance of the Language Model
from helper.models import embeddings, llm

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Create centered main title 
st.title('🦙 YouTube In-Video Search Engine')
# Create a text input box for the user
video_url = st.text_input('YouTube URL here!')

video_id = get_url_id(video_url)
# Fetch the transcript with the youtube-transcript-api
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(str(video_id))
except Exception as e:
    print(f"An error occurred: {e}")
else:
    # Convert the transcript into a Document object
    # Assuming Document requires 'text' and optionally takes 'metadata'
    documents = []
    for entry in transcript_list:
        # Calculate total seconds for start and end times
        start_total_seconds = entry['start']
        end_total_seconds = start_total_seconds + entry['duration']

        # Convert start and end total seconds into hours:minutes:seconds
        start_hours, start_minutes, start_seconds = (
            int(start_total_seconds // 3600),
            int((start_total_seconds % 3600) // 60),
            start_total_seconds % 60
        )
        end_hours, end_minutes, end_seconds = (
            int(end_total_seconds // 3600),
            int((end_total_seconds % 3600) // 60),
            end_total_seconds % 60
        )

        # Format the timestamps
        start_time_formatted = f"{start_hours:02}:{start_minutes:02}:{start_seconds:05.2f}"
        end_time_formatted = f"{end_hours:02}:{end_minutes:02}:{end_seconds:05.2f}"

        # Create the document text with the new timestamp format
        text_entry = f"{start_time_formatted} to {end_time_formatted}: {entry['text']}"

        # Create a Document object for the entry
        document = Document(text=text_entry, metadata={
            "start_hours": start_hours,
            "start_minutes": start_minutes,
            "start_seconds": start_seconds,
            "end_hours": end_hours,
            "end_minutes": end_minutes,
            "end_seconds": end_seconds
        })
        documents.append(document)

    # Now you have a list of Document objects, you can create and query an index
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    # Query can be performed as needed...

prompt = st.text_input('Prompt Example:Provide all the TimeStamps where the speaker talked about <something you are looking for>')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    response_text = response.response  # Extract the response text
    st.write(response_text)  # Display the formatted text


#Disable the below if you want to see more MetaData in the response.

# if prompt:
#     response = query_engine.query(prompt)
#     # ...and write it out to the screen
#     st.write(response)

    # Display raw response object
#    with st.expander('Response Object'):
#        st.write(response)
    # Display source text
#    with st.expander('Source Text'):
#       st.write(response.get_formatted_sources())