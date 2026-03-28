from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model= ChatAnthropic(model= "claude-3.5-sonnet-20241022", temperature= 0.8)

result= model.invoke("Who is the current Prime Minister of India")

print(result)

# print only answer
print(result.content)