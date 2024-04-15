from flask import Flask, request, jsonify, render_template
import re
from textblob import TextBlob
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# Define the rubric with corrected weights
rubric = {
    "Excellent": {
        "min_score": 90,
        "criteria": {
            "Accuracy": {"weight": 0.14},
            "Completeness": {"weight": 0.175},
            "Relevance": {"weight": 0.08},
            "Clarity": {"weight": 0.175},
            "Depth": {"weight": 0.09},
            "Organization": {"weight": 0.1},
            "Use of Evidence": {"weight": 0.09},
            "Grammar and Spelling": {"weight": 0.1},
            "Sentiment": {"weight": 0.075}
        }
    },
    "Good": {
        "min_score": 75,
        "criteria": {
            "Accuracy": {"weight": 0.14},
            "Completeness": {"weight": 0.175},
            "Relevance": {"weight": 0.08},
            "Clarity": {"weight": 0.175},
            "Depth": {"weight": 0.09},
            "Organization": {"weight": 0.1},
            "Use of Evidence": {"weight": 0.09},
            "Grammar and Spelling": {"weight": 0.1},
            "Sentiment": {"weight": 0.075}
        }
    },
    "Satisfactory": {
        "min_score": 55,
        "criteria": {
            "Accuracy": {"weight": 0.14},
            "Completeness": {"weight": 0.175},
            "Relevance": {"weight": 0.08},
            "Clarity": {"weight": 0.175},
            "Depth": {"weight": 0.09},
            "Organization": {"weight": 0.1},
            "Use of Evidence": {"weight": 0.09},
            "Grammar and Spelling": {"weight": 0.1},
            "Sentiment": {"weight": 0.075}
        }
    },
    "Fair": {
        "min_score": 40,
        "criteria": {
            "Accuracy": {"weight": 0.14},
            "Completeness": {"weight": 0.175},
            "Relevance": {"weight": 0.08},
            "Clarity": {"weight": 0.175},
            "Depth": {"weight": 0.09},
            "Organization": {"weight": 0.1},
            "Use of Evidence": {"weight": 0.09},
            "Grammar and Spelling": {"weight": 0.1},
            "Sentiment": {"weight": 0.075}
        }
    },
    "Poor": {
        "min_score": 0,
        "criteria": {
            "Accuracy": {"weight": 0.14},
            "Completeness": {"weight": 0.175},
            "Relevance": {"weight": 0.08},
            "Clarity": {"weight": 0.175},
            "Depth": {"weight": 0.09},
            "Organization": {"weight": 0.1},
            "Use of Evidence": {"weight": 0.09},
            "Grammar and Spelling": {"weight": 0.1},
            "Sentiment": {"weight": 0.075}
        }
    }
}

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    # Define the prompt for evaluation
    prompt = f"""
    Evaluate the Candidate's Response to the Following Interview Question:

    Question: '{question}'

    Evaluation Criteria:

    The following criteria will be used to assess the Candidate's Response. Briefly explain each criterion to ensure a shared understanding among evaluators.

    1. Accuracy (0-100): Is the response factually correct and aligned with industry best practices? Consider the context of the question and potential for multiple valid perspectives. (Weight: 14%)
       Explanation: An accurate response demonstrates a strong understanding of the subject matter and aligns with current industry standards. It avoids factual errors or misconceptions.

    2. Completeness (0-100): Does the response address all essential aspects of the question, including relevant details and nuances? Balance conciseness with comprehensiveness. (Weight: 17.5%)
       Explanation: A complete response covers all key points of the question without unnecessary digressions. It shows the candidate has considered various aspects of the topic.

    3. Relevance (0-100): Is the response focused on the question asked, avoiding unnecessary digressions or irrelevant information? Does it demonstrate understanding of the role's requirements? (Weight: 8%)
       Explanation: A relevant response stays on topic and connects the answer to the specific question and the demands of the role. It avoids irrelevant information or tangents.

    4. Clarity (0-100): Is the response clear, concise, and easy to understand? Does the candidate use appropriate language for a professional setting and avoid ambiguity? (Weight: 17.5%)
       Explanation: A clear response is easy to follow and avoids jargon or technical terms without explanation. It uses appropriate language for a professional setting.

    5. Depth (0-100): Does the response demonstrate a deep understanding of the topic, or is it superficial? Consider the complexity of the question and its potential for insightful analysis. (Weight: 9%)
       - For Short Answers (under 50 words): Evaluate if core concepts are addressed directly and effectively.
       - For Long Answers (over 150 words): Assess the presence of well-developed explanations, examples, or arguments that support a comprehensive understanding and problem-solving skills.
       Explanation: A deep response showcases a strong grasp of the topic beyond a surface level. It offers insights, examples, or well-developed explanations that demonstrate critical thinking and problem-solving abilities. Short answers should still be impactful and address core concepts efficiently.

    6. Organization (0-100): Is the response well-organized and structured? Does it flow logically with clear transitions, if applicable? Does it demonstrate effective communication skills? (Weight: 10%)
       Explanation: An organized response has a clear structure with a logical flow of ideas. It includes transitions between points to enhance understanding. This demonstrates effective communication skills.

    7. Use of Evidence (0-100): Does the response provide evidence (e.g., statistics, research findings, industry case studies, personal experiences) to substantiate its claims? Consider the nature of the question and whether evidence is necessary.
       - For Factual Answers: Strong evidence is crucial.
       - For Opinion-Based Answers: Evidence can add weight but isn't always mandatory. (Weight: 9%)
       Explanation: Where appropriate, the response uses relevant evidence (data, research, case studies, or personal experiences) to support its claims. This strengthens the credibility and persuasiveness of the answer.

    8. Grammar and Spelling (0-100): Evaluate the response for grammatical accuracy and spelling errors. Is it free from grammatical mistakes and typos? (Weight: 10%)
       Explanation: A response that is grammatically correct and free of spelling errors enhances readability and professionalism. It reflects the candidate's attention to detail and communication skills.

    9. Sentiment (0-100): What is the overall tone and emotion conveyed in the response? (Weight: 7.5%)
       Explanation: Assess the overall sentiment of the response, including tone and emotion. Look for professionalism, confidence, and positivity.

    Scoring & Justification:

    Assign a score (0-100) for each criterion. Justify your scores with specific examples from the Candidate's Response, highlighting both strengths and areas for improvement.
    Candidate's Response: '{answer}'
    """

    genai.configure(api_key=os.getenv("gemini_api_key"))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)

    evaluation = response.text.strip()
    scores = [float(value) for value in re.findall(r'-?\d+', evaluation)]

    breakdown = {criterion: min(score * rubric["Excellent"]["criteria"][criterion]["weight"], 100) for criterion, score in zip(rubric["Excellent"]["criteria"].keys(), scores)}

    # Calculate the final score
    final_score = min(sum(breakdown.values()), 100)

    sentiment = TextBlob(answer).sentiment.polarity

    grade = None
    for level, level_criteria in rubric.items():
        if final_score >= level_criteria["min_score"]:
            grade = level
            break

    return jsonify({
        "final_score": final_score,
        "grade": grade,
        "breakdown": breakdown,
        "sentiment": sentiment
    })

@app.route('/')
def index():
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)
