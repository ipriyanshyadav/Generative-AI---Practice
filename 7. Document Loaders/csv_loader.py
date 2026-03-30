from langchain_community.document_loaders import CSVLoader

loader= CSVLoader(
    file_path= "Social_Network_Ads.csv"
)

data= loader.load()

print(len(data))

print("\n")
print(data[1])