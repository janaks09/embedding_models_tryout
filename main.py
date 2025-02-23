from dotenv import load_dotenv
load_dotenv()
from SearchEngine import *
from DbSeeder import *


print('Dynamic context demo:')
llmProvider= input("Which provider to use for Embedding?\n1. OpenAI (code: openai)\n2. Gte (code: gte)\n3. IntFloat (code:intfloat)\n4. Universal AnglE Embedding (code:uae)\n")




# print('Data Seeding...')
# dbSeeder = DbSeeder(llmProvider)
# dbSeeder.add_context_examples_seed_data()
# print('Data Seeding Completed')


context= input("Search examples for?\n")
print("Pulling examples for {}:".format(context))
searchEngine = SearchEngine(llmProvider)
searchEngine.get_context_examples(context)