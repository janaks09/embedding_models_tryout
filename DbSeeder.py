from Embeddings import *
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

CONTEXT_EXAMPLES_COLLECTION = "context_examples"

class DbSeeder:

    def __init__(self, provider):
        self.provider = provider
        # connect to db
        connections.connect("default", host="localhost", port="19530")

    def drop_context_examples(self):
        providerSpecificTable = CONTEXT_EXAMPLES_COLLECTION + "_" + self.provider
        hasContextExamplesCollection = utility.has_collection(
            providerSpecificTable
        )
        if hasContextExamplesCollection:
            print("Drop Collection: " + providerSpecificTable)
            utility.drop_collection(providerSpecificTable)
        else:
            print("Collection `" + providerSpecificTable + "` does not exist")

    def add_context_examples_seed_data(self):

        self.drop_context_examples()

        providerSpecificTable =  CONTEXT_EXAMPLES_COLLECTION + "_" + self.provider
        embeddings = Embeddings()
        seedRecords = [
            {
                "id": 1,
                "example": "Query: What is the minimum number of sales needed to achieve the bonus threshold this quarter? Answer: MIN",
                "embeddings": [],
            },
            {
                "id": 2,
                "example": "Query: Among all our products, which one had the maximum number of complaints last year? Answer: MAX",
                "embeddings": [],
            },
            {
                "id": 3,
                "example": "Query: What's the least amount of time it has taken for a customer to get a service response? Answer: MIN",
                "embeddings": [],
            },
            {
                "id": 4,
                "example": "Query: In the company's history, what was the highest stock price ever recorded? Answer: MAX",
                "embeddings": [],
            },
            {
                "id": 5,
                "example": "Query: What is the lowest score a student can get to still pass the course? Answer: MIN",
                "embeddings": [],
            },
            {
                "id": 6,
                "example": "Query: Which employee achieved the maximum sales figures in the last fiscal year? Answer: MAX",
                "embeddings": [],
            },
            {
                "id": 7,
                "example": "Query: Who on my team had the highest total attainment in Q2 2015? Answer: DESC",
                "embeddings": [],
            },
            {
                "id": 8,
                "example": "Query: Who on my team had the lowest total attainment in Q1 2020? Answer: ASC",
                "embeddings": [],
            },
            {
                "id": 9,
                "example": "Query: Who on Mary Johnson's team had the highest total attainment in Q4 2018? Answer: DESC",
                "embeddings": [],
            },
            {
                "id": 0,
                "example": "Query: Who on Jeffrey Partyka's team had the lowest total commissions in Q3 2016? Answer: ASC",
                "embeddings": [],
            },
            {
                "id": 11,
                "example": "Query: Who on my team had the highest total attainment in Q2 2015? Answer: 1",
                "embeddings": [],
            },
            {
                "id": 12,
                "example": "Query: Who on my team had the lowest total attainment in Q1 2020? Answer: 1",
                "embeddings": [],
            },
            {
                "id": 13,
                "example": "Query: Give me my top 5 commission amounts. Answer: 5",
                "embeddings": [],
            },
            {
                "id": 14,
                "example": "Query: Who are the top 3 commission earners on Jeffrey Partyka's team in Q3 2016? Answer: 3",
                "embeddings": [],
            },
            {
                "id": 15,
                "example": "Query: Who are the 10 worst commission earners on Jeffrey Partyka's team in Q3 2019? Answer: 10",
                "embeddings": [],
            },
        ]

        for seedRecord in seedRecords:
            print(seedRecord["example"])
            seedRecord["embeddings"] = embeddings.get_embeddings(seedRecord["example"], self.provider)
            
        hasContextExamplesCollection = utility.has_collection(
            providerSpecificTable
        )
        print(
            f"Does collection {providerSpecificTable} exist in Milvus: {hasContextExamplesCollection}"
        )

        # create collection
        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="example", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embeddings.get_embeddingModelInfo(self.provider).dimension),
        ]

        schema = CollectionSchema(
            fields, f"{providerSpecificTable} contains examples of different context"
        )

        print(f"Collection{providerSpecificTable} created successfully\n")
        context_examples = Collection(
            providerSpecificTable, schema, consistency_level="Strong"
        )

        # insert data
        print("Start inserting seed data..")
        insert_result = context_examples.insert(seedRecords)
        context_examples.flush()
        print(
            f"Total number of records in {providerSpecificTable}: {context_examples.num_entities}"
        )

        # create index
        print("Start creating index IVF_FLAT")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        }
        context_examples.create_index("embeddings", index_params)

        print(f"{providerSpecificTable} seeding completed.")
