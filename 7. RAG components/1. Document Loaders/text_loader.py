from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Define the model
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational"
)

model= ChatHuggingFace(llm= llm)

parser= StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

loader= TextLoader('cricket.txt', encoding='utf-8')

docs= loader.load()

# print(docs)
print(type(docs))

print("\n")
print(len(docs))

print("\n")
print(docs[0].page_content)

print("\n")
print(docs[0].metadata)

print("\n")
chain= prompt | model | parser
print(chain.invoke({'poem': docs[0].page_content}))