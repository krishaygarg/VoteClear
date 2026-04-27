from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import pickle
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
load_dotenv()
class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
with open('policyareas.pkl', 'rb') as f: PREDEFINED_POLICY_AREAS = pickle.load(f)
with open('candidates.pkl', 'rb') as f: confirmed_candidates = pickle.load(f)
with open('research.pkl', 'rb') as f: all_candidates_research_data = pickle.load(f)

def run_mock_election(all_candidates_research_data, confirmed_candidates, PREDEFINED_POLICY_AREAS):
    question_history = []  # store tuples: (question, answer)

    # Step 1: Go through each policy area
    for j, area in enumerate(PREDEFINED_POLICY_AREAS):
        candidate_positions = [all_candidates_research_data[i][j] for i in range(len(confirmed_candidates))]
        prompt = ChatPromptTemplate.from_template("""
        You are designing a practical, easy-to-answer mock election quiz.
        Policy area: {area}
        Candidate positions:
        {candidate_positions}

        Task:
        1. Decide if it is useful to ask a question on this area. If not, respond 'SKIP'.
        2. If yes, generate ONE practical multiple choice question based on everyday life that a user can answer.
           - Practical means something directly related to the user’s daily experiences and choices.
           - Avoid abstract or theoretical questions like political philosophy or ideology.
           - For each option you provide, give a  explanation how this could look in practice. What would be the pros and cons for people, and how would each option affect their lives? 
                The explanation for each option should be about 3 sentences that are valuable for the user to understand. Make sure it doesn't create bias towards a certain option. 
                                                  Clear pros and cons for each option should be listed so users can understand the nuances of each choice.
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

    # Step 2: Adaptive follow-up questions (up to 5), one at a time
    print("\n--- Adaptive Follow-up Questions ---\n")
    for _ in range(3):
        prompt_followup = ChatPromptTemplate.from_template("""
You are running an adaptive round of a mock election quiz.

Candidate positions:
{candidate_info}

Previous user answers:
{previous_answers}

Task:
- Your goal is to better understand the user's real-world preferences and priorities so that their views can be compared with the candidate positions. 
- Ask at most ONE additional question, and only if:
  1. It would reveal new information about the user that cannot be reasonably inferred from their previous answers.  
  2. The question is practical — meaning it connects to the user’s everyday experiences, decisions, or concerns (e.g., costs, convenience, access, quality of services, lifestyle impacts).  
- Do not repeat or rephrase topics the user has already addressed.  
- If no further useful question remains, respond only with 'NONE'.
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

    # Step 3: Holistic review
    prompt_review = ChatPromptTemplate.from_template("""
    Write a holistic review of how the user’s answers align with each candidate.
    Candidate positions: {candidate_info}
    User responses: {previous_answers}

    - Summarize where the user’s preferences match each candidate.
    - Highlight tradeoffs in practical, everyday terms.
    - Make it descriptive and easy to understand.
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
run_mock_election(all_candidates_research_data, confirmed_candidates, PREDEFINED_POLICY_AREAS)