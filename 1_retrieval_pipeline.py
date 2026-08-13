from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# Load vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="dbv2/chroma_db", embedding_function=embedding_model)

# Retrival function
def retrieve_chunks(query, k=35): 
    retriever = db.as_retriever(search_kwargs={"k": k}) 
    return retriever.invoke(query)


# Build context
def build_context(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

# LLM call function
def ask_llm(query, context):
    
    prompt = f"""
You are extracting evidence for a pre-earthquake visual vulnerability
assessment database from earthquake reconnaissance reports.

User Request:
{query}

TASK:
Identify specific, independently assessable building or site characteristics
that are explicitly described in the context, and record any earthquake
damage explicitly associated with each characteristic.

IMPORTANT:

A parameter must be ONE specific observable building/site characteristic.

If one sentence contains multiple characteristics, SPLIT them into separate
parameters.

Example:
"Extensive damage because of heavy roof load, lack of roof trusses and
lack of wall plates"

Extract:
• Heavy Roof Load
• Lack of Roof Trusses
• Lack of Wall Plates

Do NOT combine multiple characteristics into one parameter.

Extract ONLY characteristics that:
• are explicitly present in the text
• can reasonably be identified through visual/field inspection
• describe a specific condition, configuration, absence, presence, or feature

Do NOT infer vulnerability or engineering significance.

Do NOT extract:
• generic terms such as "wall", "foundation", "masonry", "roof", or "building"
  unless a specific characteristic is described
• strength, stiffness, capacity, or other engineering properties
• hidden structural information
• design/calculation-based information
• laboratory-tested material properties
• building names or locations
• damage alone without an associated characteristic
• repair or retrofit measures

IMPORTANT:
Do not treat a building material or typology as a vulnerability parameter
unless the text explicitly associates that characteristic with damage or
describes it as a relevant building condition.

Examples:

"thick mud masonry walls ... partial or total collapse"
→ Parameter: Thick Mud Masonry Walls
→ Damage: Partial or Total Collapse

"houses on steep slopes have greater damage"
→ Parameter: Steep-Slope Location
→ Damage: Greater Damage

"shear cracks developed in walls"
→ Do not extract a parameter unless an associated characteristic is explicitly
  stated.

"heavy roof load, lack of roof trusses and wall plates caused extensive damage"
→ Parameter 1: Heavy Roof Load
→ Parameter 2: Lack of Roof Trusses
→ Parameter 3: Lack of Wall Plates

Context: {context}

OUTPUT:Return ONLY a numbered list:

1. Parameter: ...
   Damage: ...
   Evidence: "Exact supporting statement from the context"

Rules:
• Each numbered item must contain ONE parameter only.
• Evidence must be copied exactly from the context.
• Do not invent or paraphrase evidence.
• Do not create causal relationships that are not explicitly stated.
• If damage is not explicitly associated with a parameter:
  Damage: Not explicitly reported
• Remove exact duplicate parameters within the retrieved context.
"""


    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

    query ="""
    Extract observable building and site characteristics and their explicitly
    documented earthquake-damage associations for Stone Masonry buildings.
    """
    result = ask(query)
    print("\n==================== FINAL ANSWER ====================\n")
    print(result)   

    with open("pipeline1_evidence.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("\nPipeline 1 evidence saved.")