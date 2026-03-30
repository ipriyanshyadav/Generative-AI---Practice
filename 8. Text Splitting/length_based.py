from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader(
    file_path= "dl-curriculum.pdf"
)

data= loader.load()

splitter= CharacterTextSplitter(
    chunk_size= 200,
    chunk_overlap= 20,
    separator= ''
)

result= splitter.split_documents(data)
print(result[0].page_content)

print("\n")
print(result[0].metadata)