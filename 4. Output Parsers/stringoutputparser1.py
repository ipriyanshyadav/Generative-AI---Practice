from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the model
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational"
)

model= ChatHuggingFace(llm= llm)

# 1st prompt
template1= PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt
template2= PromptTemplate(
    template='Write 5 point summary in the following text.\n {text}',
    input_variables=['text']
)

parser= StrOutputParser()

chain= template1 | model | template2 | model | parser

result= chain.invoke({'topic':"Black Hole"})

print(result)