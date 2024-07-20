from pymilvus import model
from pymilvus import MilvusClient
from pymilvus import model
from pymilvus import MilvusClient

class MilvusDemo:
    def __init__(self, collection_name):
        self.embedding_fn = model.DefaultEmbeddingFunction() # "paraphrase-albert-small-v2" (~50MB).
        self.client = MilvusClient("milvus_demo.db")
        self.collection_name = collection_name


    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        status = self.client.create_collection(collection_name=self.collection_name, dimension=768)
        return status
    
    
    def insert_docs(self, docs=[]):
        vectors = self.embedding_fn.encode_documents(docs)
        print("Dim:", self.embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)
        data = [
            {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
            for i in range(len(vectors)) 
        ]
        status = self.client.insert(collection_name=self.collection_name, data=data)
        return status


    def search_collection(self, input_queries=["Who is Alan Turing?"]):
        query_vectors = self.embedding_fn.encode_queries(input_queries)
        res = self.client.search(
            collection_name=self.collection_name,  # target collection
            data=query_vectors,  # query vectors
            filter="subject == 'biology'",
            limit=2,  # number of returned entities
            output_fields=["text", "subject"],  # specifies fields to be returned
        )
        return res
    

    def query_demo_collection(self):
        res = self.client.query(
            collection_name=self.collection_name,
            filter="subject == 'history'",
            output_fields=["text", "subject"],
        )
        return res

##
# # load existing
# from pymilvus import MilvusClient
# client = MilvusClient("milvus_demo.db")
# client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]
mvobj = MilvusDemo("mytestcoll")
st = mvobj.create_collection()
print(st)
st = mvobj.insert_docs(docs)
print(st)
