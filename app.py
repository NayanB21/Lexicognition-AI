import streamlit as st
import tempfile
import re

from core.pdf_parser import extract_text_from_pdf
from core.chunking import semantic_chunk_text_v2
from core.embeddings import build_faiss_index
from core.retrieval import retrieve_chunks
from core.question_agent import generate_viva_questions
from core.evaluation_agent import evaluate_answer

def format_evaluation(text: str):
    score = re.search(r"Score:\s*(\d+)", text)
    verdict = re.search(r"Verdict:\s*([A-Za-z ]+)", text)
    explanation = re.search(r"Explanation:\s*(.*)", text, re.DOTALL)

    return {
        "score": score.group(1) if score else "N/A",
        "verdict": verdict.group(1).strip() if verdict else "N/A",
        "explanation": explanation.group(1).strip() if explanation else text
    }

def extract_score(evaluation_text):
    match = re.search(r"Score:\s*(\d+)", evaluation_text)
    return int(match.group(1)) if match else 0

# ----------------------------------
# Session State Initialization
# ----------------------------------
if "semantic_chunks" not in st.session_state:
    st.session_state.semantic_chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "questions" not in st.session_state:
    st.session_state.questions = None

if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0

if "viva_started" not in st.session_state:
    st.session_state.viva_started = False
if "history" not in st.session_state:
    st.session_state.history = []


st.set_page_config(page_title="Lexicognition AI", layout="centered")
st.title("🎓 Lexicognition AI – Viva Voce Examiner")


# ----------------------------------
# PDF Upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully")

    if st.button("Start Viva"):
        with st.spinner("Analyzing paper..."):
            paper_text = extract_text_from_pdf(pdf_path)

            # Semantic chunking
            st.session_state.semantic_chunks = semantic_chunk_text_v2(paper_text)

            # Build FAISS index
            st.session_state.index, _ = build_faiss_index(
                st.session_state.semantic_chunks
            )

            # Retrieve strong context for question generation
            question_themes = [
                "problem motivation and limitations of existing approaches",
                "core architecture and design decisions",
                "training methodology and optimization strategy",
                "advantages and trade-offs of the proposed method",
                "limitations, assumptions, and failure cases"
            ]

            all_retrieved = []

            for theme in question_themes:
                chunks = retrieve_chunks(
                    theme,
                    st.session_state.index,
                    st.session_state.semantic_chunks,
                    top_k=2
                )
                all_retrieved.extend(chunks)

            questions_text = generate_viva_questions(all_retrieved)



            st.session_state.questions = [
                q.strip()
                for q in re.findall(r"\d+\.\s*(.+)", questions_text)
            ]

            st.session_state.current_q_idx = 0
            st.session_state.viva_started = True


# ----------------------------------
# Viva Loop
# ----------------------------------
if st.session_state.viva_started and st.session_state.questions:

# -----------------------------
# Render Previous Q&A History
# -----------------------------
    if st.session_state.history:
        st.markdown("## 🧾 Viva Progress")

        for i, item in enumerate(st.session_state.history):
            st.markdown(f"### Question {i+1}")
            st.write(item["question"])

            st.markdown("**Your Answer:**")
            st.write(item["answer"])

            st.markdown("**Evaluation:**")
            formatted = format_evaluation(item["evaluation"])

            st.markdown(f"**Score:** {formatted['score']}")
            st.markdown(f"**Verdict:** {formatted['verdict']}")
            st.markdown("**Explanation:**")
            st.write(formatted["explanation"])
            st.markdown("---")
        total_score = sum(item["score"] for item in st.session_state.history)
        st.markdown("Total score till now:")
        st.write(total_score)



    q_idx = st.session_state.current_q_idx
    questions = st.session_state.questions

    if q_idx < len(questions):
        st.markdown("## 📝 Current Question")
        st.subheader(f"Question {q_idx + 1}")
        st.write(questions[q_idx])

        answer = st.text_area(
            "Your Answer",
            key=f"answer_{q_idx}"
        )

        if st.button("Submit Answer", key=f"submit_{q_idx}"):

            if len(answer.strip()) < 10:
                st.warning("Please write a meaningful answer.")
                st.stop()

            retrieved = retrieve_chunks(
                questions[q_idx],
                st.session_state.index,
                st.session_state.semantic_chunks,
                top_k=3
            )

            evaluation = evaluate_answer(
                questions[q_idx],
                answer,
                retrieved
            )
            score = extract_score(evaluation)
            st.session_state.history.append({
                "question": questions[q_idx],
                "answer": answer,
                "evaluation": evaluation,
                "score": score
            })

            # 🔑 THIS is what moves to next question
            st.session_state.current_q_idx += 1
            st.rerun()


    else:
        # st.success("🎉 Viva Completed (All questions asked)")
        final_score = sum(item["score"] for item in st.session_state.history)
        st.success(f"🎉 Viva Completed! Final Score: {final_score} / {len(questions)*10}")

