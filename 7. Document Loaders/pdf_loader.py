from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('dl-curriculum.pdf')

docs= loader.load()

print(len(docs))

print("\n")
print(docs[0].page_content)

print("\n")
print(docs[0].metadata)