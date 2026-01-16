from core.llm import gemini_llm


def evaluate_answer(question, user_answer, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict university viva examiner.

You are evaluating a student's answer to a research paper question.

QUESTION:
{question}

STUDENT ANSWER:
{user_answer}

REFERENCE CONTEXT (from the paper):
----------------
{context}
----------------

EVALUATION RULES:
- Judge ONLY using the provided context.
- If the answer is factually wrong or misleading, mark it low.
- If partially correct, explain what is missing.
- If correct but shallow, penalize depth.
- If correct and deep, reward it.
- Be strict. Do NOT be generous.

OUTPUT FORMAT (STRICT):
Score: <number between 1 and 10>
Verdict: <Correct / Partially Correct / Incorrect>
Explanation: <2–3 lines max>
"""

    return gemini_llm(prompt)
