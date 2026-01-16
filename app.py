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

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "questions" not in st.session_state:
    st.session_state.questions = None

if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0

if "viva_started" not in st.session_state:
    st.session_state.viva_started = False
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------
st.set_page_config(
    page_title="Lexicognition AI",
    layout="wide"
)

st.sidebar.title("🎓 Lexicognition AI")
st.sidebar.markdown("AI-powered Viva Examiner")


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


# ---------------
# Viva Loop
# ---------------
if st.session_state.viva_started and st.session_state.questions:

    if st.session_state.viva_started:
        if st.sidebar.button("📜 Toggle History"):
            st.session_state.show_history = not st.session_state.show_history

# -----------------------------
# Render Previous Q&A History
# -----------------------------
    if st.session_state.show_history and st.session_state.history:
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
    if f"show_answer_{q_idx}" not in st.session_state:
        st.session_state[f"show_answer_{q_idx}"] = False

    if f"show_eval_{q_idx}" not in st.session_state:
        st.session_state[f"show_eval_{q_idx}"] = False

    if f"answered_{q_idx}" not in st.session_state:
        st.session_state[f"answered_{q_idx}"] = False



    if q_idx < len(questions):
        st.sidebar.progress(
            q_idx / len(questions)
        )
        st.sidebar.markdown(
            f"**Question:** {q_idx} / {len(questions)}"
        )
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
            st.session_state[f"answered_{q_idx}"] = True
            st.session_state[f"evaluation_{q_idx}"] = evaluation
            st.session_state[f"score_{q_idx}"] = score

            st.session_state.history.append({
                "question": questions[q_idx],
                "answer": answer,
                "evaluation": evaluation,
                "score": score
            })
            st.rerun()



        if st.session_state[f"answered_{q_idx}"]:
        
            evaluation = st.session_state[f"evaluation_{q_idx}"]

            col1, col2 = st.columns(2)

            with col1:
                if st.button("👁️ Show My Answer", key=f"btn_answer_{q_idx}"):
                    st.session_state[f"show_answer_{q_idx}"] ^= True
                    st.session_state[f"show_eval_{q_idx}"] = False

            with col2:
                if st.button("🧪 Show Evaluation", key=f"btn_eval_{q_idx}"):
                    st.session_state[f"show_eval_{q_idx}"] ^= True
                    st.session_state[f"show_answer_{q_idx}"] = False

            col_ans, col_eval = st.columns(2)

            if st.session_state[f"show_answer_{q_idx}"]:
                with col_ans:
                    st.markdown("### 🧾 Your Answer")
                    st.write(answer)

            if st.session_state[f"show_eval_{q_idx}"]:
                with col_eval:
                    st.markdown("### 🧪 Evaluation")
                    st.markdown("---")
                    formatted = format_evaluation(evaluation)
                    st.metric("Score", formatted["score"])
                    st.markdown(f"**Verdict:** {formatted['verdict']}")
                    st.markdown("**Explanation:**")
                    st.write(formatted["explanation"])


            # moves to next question

            if st.session_state[f"answered_{q_idx}"]:
                st.markdown("---")

                col_prev, col_next = st.columns([1, 1])

                with col_next:
                    if st.button("➡️ Next Question", key=f"next_{q_idx}"):
                        st.session_state[f"show_answer_{q_idx}"] = False
                        st.session_state[f"show_eval_{q_idx}"] = False
                        st.session_state.current_q_idx += 1
                        st.rerun()


    else:
        # st.success("🎉 Viva Completed (All questions asked)")
        final_score = sum(item["score"] for item in st.session_state.history)
        st.sidebar.metric("Total Score", final_score)
        st.success(f"🎉 Viva Completed! Final Score: {final_score} / {len(questions)*10}")

