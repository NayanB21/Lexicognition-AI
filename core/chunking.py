import re

def semantic_chunk_text_v2(text, max_chunk_size=900, overlap=150):
    # Normalize text
    text = re.sub(r'\s+', ' ', text).strip()

    # Split by strong paragraph boundaries
    paragraphs = re.split(r'\n{2,}|\.\s(?=[A-Z])', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue  # ignore noise

        if len(current_chunk) + len(para) <= max_chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks