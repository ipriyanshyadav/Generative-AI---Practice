from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the model
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational"
)

model= ChatHuggingFace(llm= llm)

prompt= PromptTemplate(
    template= "write a joke on {topic}",
    input_variables= ['topic']
)

prompt2= PromptTemplate(
    template= "explain the joke in 2 lines {text}",
    input_variables= ['text']
)

def wrap_text(text):
    return {'text': text}

parser= StrOutputParser()

chain= RunnableSequence(prompt, model, parser, prompt2, model, parser)
print(chain.invoke({'topic': 'AI'}))