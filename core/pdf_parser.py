import fitz

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right

        for block in blocks:
            text = block[4]
            if len(text.strip()) > 20:
                full_text.append(text.strip())

    return "\n".join(full_text)
