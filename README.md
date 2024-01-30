# DSlogic: Data Science Assistant

DSlogic is your open-source, LLM-powered assistant, guiding you through every step of your DS journey. Imagine having a wise mentor whispering best practices in your ear, from data cleaning to model selection and probably more. 

Try it: https://huggingface.co/spaces/inmind/DSlogic
#
### Architecture
![LLM app](https://github.com/SwaroopMeher/DSlogic/assets/115743490/cde46f5e-6387-4ffa-9b4d-3d83297ff4a5)
#
### Under the hood:
- Frontend: Streamlit
- Backend: FastAPI
- Vector Database: Qdrant
- LLM Model: Mixtral-8x7B-Instruct-v0.1
- Containerization: Docker
- Deployment: Hugging Face Spaces

#

### How to run locally
1. Clone the repo
2. Add your keys in a config.ini file in the below manner.
```
[Qdrant]
api_key = 
url = 

[HuggingFaceHub]
api_token = 
```
4. ```streamlit run main.py```

#
### Contribution guidelines

Coming soon
