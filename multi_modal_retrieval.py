from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Load vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="dbv2/chroma_db",
    embedding_function=embedding_model)


# Retrival function
def retrieve_chunks(query, k=5):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)



# Build context
def build_context(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

# LLM call function
def ask_llm(query, context):
    
    prompt = f"""
You are an earthquake engineering expert.

QUESTION:
{query}

CONTEXT:
{context}

Task:
Extract all building vulnerability parameters mentioned in the context.

Rules:
- Return ONLY parameter names.
- One parameter per line.
- Remove duplicates.
- Do not explain anything.
- Do not categorize.
- Do not infer or add parameters that are not mentioned.
- If no parameters are found, return: No vulnerability parameters found.

Example Output:


1.Number of Storeys
2.Wall Thickness
3.Roof Type
4.Plan Irregularity

"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=,
                                 temperature=0)
    
    response = llm.invoke(prompt)

    return response.content

# main function
def ask(query):

    # Step 1: retrieve relevant chunks
    chunks = retrieve_chunks(query) 

    # Step 2: build context
    context = build_context(chunks)

    # Step 3: get LLM response
    answer = ask_llm(query, context)

    return answer


if __name__ == "__main__":

    query ="Identify all earthquake vulnerability parameters mentioned for random rubble stone masonry buildings in the provided documents."

    result = ask(query)

    print("\n==================== FINAL ANSWER ====================\n")
    print(result)