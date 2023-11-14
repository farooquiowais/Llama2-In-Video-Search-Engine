from transformers import AutoTokenizer
from secrets import auth_token
import torch
import re

# Define variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-7b-chat-hf"


def get_url_id(full_url:str):
  pattern = r"v=([^&]+)"

  # Search for the pattern in the URL
  match = re.search(pattern, full_url)

  # Extract the matched part
  video_id = match.group(1) if match else None
  return str(video_id)


@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , use_auth_token=auth_token, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True) 

    return tokenizer, model