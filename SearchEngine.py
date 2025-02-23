from pymilvus import (
    connections,
    Collection,
)
from Embeddings import *

CONTEXT_EXAMPLES_COLLECTION = 'context_examples'

class SearchEngine:
    
    def __init__(self, provider):
        self.provider = provider
        #connect to db
        connections.connect("default", host="localhost", port="19530")
        
    def get_context_examples(self, contextText):
        
        # get embeddings
        embeddings = Embeddings()
        query_embeddings = embeddings.get_embeddings(contextText, self.provider)
        
        # performing search
        print("Loaidng context_examples in inmemory")
        providerSpecificTable =  CONTEXT_EXAMPLES_COLLECTION + "_" + self.provider
        context_examples = Collection(providerSpecificTable)
        context_examples.load()
        
        # perform similarity search
        print("Performing vector similarity search")
        search_params = {
            "metric_type": "COSINE"
        }
        
        search_result = context_examples.search(data = [query_embeddings], anns_field = "embeddings", param = search_params, limit=4, output_fields=["example"], consistency_level="Strong")
        
        print("Search Results")
        for hits in search_result:
            for hit in hits:
                print(f"{hit.entity.get('example')} -- Distance:{hit.distance} -- Score:{hit.score}")
        
        context_examples.release()
        