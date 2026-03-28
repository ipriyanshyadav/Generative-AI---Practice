from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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

prompt1= template1.invoke({'topic':"Black Hole"})
result= model.invoke(prompt1)

prompt2= template2.invoke({'text':result})
result1= model.invoke(prompt2)

print(result1.content)