import os
import base64
# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

IMAGE_FOLDER = "dbv2/extracted_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def extract_and_save_images(elements, source_file):
    """Extract all images from PDF elements and save them as PNG."""

    pdf_name = os.path.splitext(source_file)[0]
    image_count = 1

    for element in elements:

        if type(element).__name__ != "Image":
            continue
        if not hasattr(element, "metadata"):
            continue
        if not hasattr(element.metadata, "image_base64"):
            continue

        image_bytes = base64.b64decode(element.metadata.image_base64)

        image_path = os.path.join(IMAGE_FOLDER, f"{pdf_name}_image_{image_count:04d}.png")

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        image_count += 1

    return image_count - 1


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path, # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"],# Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )
    images_saved = extract_and_save_images(elements, os.path.basename(file_path))
    
    print(f"Extracted {len(elements)} elements")
    return elements, images_saved
    

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    """Extract text and tables in structured format"""

    content_data = {"text": chunk.text, "tables": [], "types": {"text"}}

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):

        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # ---------------- TABLE ----------------
            if element_type == "Table":
                content_data["types"].add("table")

                table_html = getattr(element.metadata, "text_as_html", element.text)

                content_data["tables"].append({"html": table_html, "text": table_html})

    # convert set → list for JSON compatibility
    content_data["types"] = list(content_data["types"])

    return content_data


def summarise_chunks(chunks, source_file=None):
    """
    Convert chunks into LangChain Documents for RAG.
    Only prepares data (NO DB operations here).
    """
    langchain_documents = []

    for i, chunk in enumerate(chunks):

        # Step 1: Extract structured content
        content_data = separate_content_types(chunk)

        # Step 2: Build searchable text
        searchable_content = content_data["text"]

        if content_data["tables"]:
            table_texts = [t["text"] for t in content_data["tables"]]
            searchable_content += "\n\n[TABLE CONTEXT]\n" + "\n\n".join(table_texts)

        # Step 3: Create document metadata
        metadata = {
            "source_file": source_file,
            "chunk_id": i,
            "has_table": len(content_data["tables"]) > 0,
            "table_count": len(content_data["tables"])
        }

        # Step 4: Create LangChain Document
        doc = Document(page_content=searchable_content, metadata=metadata)

        langchain_documents.append(doc)

    return langchain_documents

def create_vector_store(documents, persist_directory="dbv2/chroma_db"):
    """Create and persist ChromaDB vector store"""

    print("Creating embeddings and storing in ChromaDB...")
    print(f"Total documents: {len(documents)}")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector store created and saved successfully")
    return vectorstore

def run_complete_ingestion_pipeline(folder_path: str):
    """Process all PDFs in a folder and create ONE vector database"""

    print("Starting Multi-PDF RAG Pipeline")
    print("=" * 50)

    all_documents = []
    total_images = 0

    pdf_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in folder")
        return None

    print(f"Found {len(pdf_files)} PDF files")

    for i, pdf_file in enumerate(pdf_files):

        print(f"\nProcessing {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")

        try:
            elements, image_count = partition_document(pdf_file)
            total_images += image_count
            chunks = create_chunks_by_title(elements)
            documents = summarise_chunks(chunks, source_file=os.path.basename(pdf_file))
            all_documents.extend(documents)

        except Exception as e:
            print(f" Error processing {pdf_file}: {e}")
            continue

    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total PDFs processed : {len(pdf_files)}")
    print(f"Total chunks created : {len(all_documents)}")
    print(f"Total images saved   : {total_images}")
    print("=" * 50)

    db = create_vector_store(all_documents, persist_directory="dbv2/chroma_db")

    print("🎉 Multi-PDF Pipeline completed successfully!")

    return db

# Run the complete pipeline
db = run_complete_ingestion_pipeline(r"C:\RAG_for_EQ\docs")