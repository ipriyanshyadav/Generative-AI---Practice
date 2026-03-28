from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model= ChatOpenAI(model= "gpt-4", temperature= 0.8, max_tokens= 10)

result= model.invoke("Who is the current Prime Minister of India")

print(result)

# print only answer
print(result.content)