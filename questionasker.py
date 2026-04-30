import pickle

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

with open('policyareas.pkl', 'rb') as f:
    PREDEFINED_POLICY_AREAS = pickle.load(f)
with open('candidates.pkl', 'rb') as f:
    confirmed_candidates = pickle.load(f)
with open('research.pkl', 'rb') as f:
    all_candidates_research_data = pickle.load(f)

def run_mock_election(all_candidates_research_data, confirmed_candidates, PREDEFINED_POLICY_AREAS):
    question_history = []

    for j, area in enumerate(PREDEFINED_POLICY_AREAS):
        candidate_positions = [all_candidates_research_data[i][j] for i in range(len(confirmed_candidates))]
        prompt = ChatPromptTemplate.from_template("""
        You are designing a practical, easy-to-answer mock election quiz.
        Policy area: {area}
        Candidate positions:
        {candidate_positions}

        Task:
        1. Decide if it is useful to ask a question on this area. If not, respond 'SKIP'.
        2. If yes, generate ONE practical multiple choice question based on everyday life.
           - Provide balanced pros and cons for each option (about 3 sentences).
        """)

        messages = prompt.format_messages(
            area=area,
            candidate_positions="\n".join([f"{confirmed_candidates[i]}: {candidate_positions[i].summary}"
                                             for i in range(len(confirmed_candidates))])
        )
        response = chat.invoke(messages).content.strip()

        if response != "SKIP":
            print(response)
            answer = input("Your answer: ")
            question_history.append((response, answer))

    print("\n--- Adaptive Follow-up Questions ---\n")
    for _ in range(3):
        prompt_followup = ChatPromptTemplate.from_template("""
You are running an adaptive round of a mock election quiz.

Candidate positions:
{candidate_info}

Previous user answers:
{previous_answers}

Task:
- Ask at most ONE additional practical question that reveals new information.
- Do not repeat topics already addressed.
- If no further useful question remains, respond with 'NONE'.
        """)

        candidate_info = "\n\n".join([
            f"{c}:\n" + "\n".join([f"{area}: {all_candidates_research_data[i][j].summary}"
                                         for j, area in enumerate(PREDEFINED_POLICY_AREAS)])
            for i, c in enumerate(confirmed_candidates)
        ])

        previous_answers_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in question_history])

        followup_question = chat.invoke(
            prompt_followup.format(
                candidate_info=candidate_info,
                previous_answers=previous_answers_text
            )
        ).content.strip()

        if followup_question == "NONE":
            break

        print(followup_question)
        answer = input("Your answer: ")
        question_history.append((followup_question, answer))

    prompt_review = ChatPromptTemplate.from_template("""
    Write a holistic review of how the user's answers align with each candidate.
    Candidate positions: {candidate_info}
    User responses: {previous_answers}

    - Summarize where the user's preferences match each candidate.
    - Highlight tradeoffs in practical, everyday terms.
    - Avoid numeric scores or recommending a candidate.
    """)

    review = chat.invoke(
        prompt_review.format(
            candidate_info=candidate_info,
            previous_answers=previous_answers_text
        )
    ).content.strip()
    print("\n=== Holistic Review ===")
    print(review)


if __name__ == "__main__":
    run_mock_election(all_candidates_research_data, confirmed_candidates, PREDEFINED_POLICY_AREAS)