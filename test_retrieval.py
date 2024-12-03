import chromadb


# Initialize Chroma Client
chroma_client = chromadb.PersistentClient(path="db/")

# Retrieve the collection
try:
    collection = chroma_client.get_collection(name="icici_etf_articles")
except chromadb.errors.InvalidCollectionException:
    print(
        "Collection 'icici_etf_articles' does not exist. Please check your database setup."
    )
    exit()


# Query the Database
def query_database(user_query, n_results=5):
    """Query the Chroma database with a user query."""
    results = collection.query(
        query_texts=[user_query],  # The query string
        n_results=n_results,  # Number of results to retrieve
    )
    return results


# Example Query
if __name__ == "__main__":
    query = "Guide me sector etfs please?"  # Replace with your query
    results = query_database(query)

    # print(results)
    print("Distances:", results["distances"][0])
    print("Titles:", results["metadatas"][0])
    # print("Documents", results["documents"][0])
    print(results)
    # Get the total number of chunks by querying the collection for all documents
    all_documents = collection.get()
    num_chunks = len(all_documents["documents"])
