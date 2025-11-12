"""
ML/AI Interview Practice Bot
A simple app to practice interview questions and get AI feedback
"""

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
from questions import get_all_questions, get_questions_by_category, QUESTIONS

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Please check your .env file.")
    st.stop()
    
genai.configure(api_key=GEMINI_API_KEY)

# Page config
st.set_page_config(
    page_title="ML Interview Practice Bot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'user_answer' not in st.session_state:
    st.session_state.user_answer = ""
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'questions_answered' not in st.session_state:
    st.session_state.questions_answered = 0
if 'category_stats' not in st.session_state:
    st.session_state.category_stats = {category: 0 for category in QUESTIONS.keys()}
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "All Categories"

def get_random_question(category="All Categories"):
    """Get a random question from the question bank"""
    if category == "All Categories":
        all_questions = get_all_questions()
    else:
        all_questions = []
        questions = get_questions_by_category(category)
        for q in questions:
            q_copy = q.copy()
            q_copy['category'] = category
            all_questions.append(q_copy)
    
    return random.choice(all_questions) if all_questions else None

def get_feedback(question, user_answer):
    """Get AI feedback on the user's answer"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are an experienced ML/AI interviewer helping a candidate practice for internship interviews.

Question: {question['question']}

Key points that should be covered:
{', '.join(question['key_points'])}

Candidate's answer:
{user_answer}

Please provide constructive feedback on their answer. Include:
1. What they did well
2. What key points they missed (if any)
3. Suggestions for improvement
4. A rating out of 5 (format: "Rating: X/5")

Keep the feedback encouraging and specific. Remember, this is practice for an internship position, so adjust expectations accordingly."""

        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error getting feedback: {str(e)}"

# App UI
st.title("🤖 ML Interview Practice Bot")
st.markdown("Practice ML/AI interview questions and get instant feedback!")

# Sidebar with stats and category filter
with st.sidebar:
    st.header("📊 Your Progress")
    st.metric("Total Questions Answered", st.session_state.questions_answered)
    
    st.markdown("---")
    st.subheader("Progress by Category")
    for category, count in st.session_state.category_stats.items():
        st.markdown(f"**{category}:** {count} questions")
    
    st.markdown("---")
    st.subheader("🎯 Practice Mode")
    
    categories = ["All Categories"] + list(QUESTIONS.keys())
    selected = st.selectbox(
        "Choose a category:",
        categories,
        index=categories.index(st.session_state.selected_category)
    )
    
    if selected != st.session_state.selected_category:
        st.session_state.selected_category = selected
        st.session_state.current_question = None
        st.session_state.feedback = None
        st.session_state.user_answer = ""
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("Take your time to think through your answer. Quality > Speed!")
    
    if st.button("🔄 Reset Progress"):
        st.session_state.questions_answered = 0
        st.session_state.category_stats = {category: 0 for category in QUESTIONS.keys()}
        st.session_state.current_question = None
        st.session_state.feedback = None
        st.session_state.user_answer = ""
        st.rerun()

# Main content area
if st.session_state.current_question is None:
    st.info(f"👋 Ready to practice? Click below to get a question from **{st.session_state.selected_category}**!")
    if st.button("Get Question", type="primary"):
        question = get_random_question(st.session_state.selected_category)
        if question:
            st.session_state.current_question = question
            st.session_state.feedback = None
            st.session_state.user_answer = ""
            st.rerun()
        else:
            st.error("No questions available for this category!")
else:
    # Display current question
    question = st.session_state.current_question
    
    st.subheader(f"📝 {question['category']}")
    st.markdown(f"### {question['question']}")
    
    # Show hints in an expander
    with st.expander("💡 Need a hint?"):
        for hint in question['hints']:
            st.markdown(f"• {hint}")
    
    # Answer input
    user_answer = st.text_area(
        "Your Answer:",
        value=st.session_state.user_answer,
        height=200,
        placeholder="Type your answer here... Take your time and be thorough!",
        key="answer_input"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Submit Answer", type="primary", disabled=st.session_state.feedback is not None):
            if user_answer.strip():
                with st.spinner("Getting feedback..."):
                    feedback = get_feedback(question, user_answer)
                    st.session_state.feedback = feedback
                    st.session_state.questions_answered += 1
                    st.session_state.category_stats[question['category']] += 1
                    st.session_state.user_answer = user_answer
                    st.rerun()
            else:
                st.warning("Please write an answer before submitting!")
    
    with col2:
        if st.button("Skip Question"):
            st.session_state.current_question = get_random_question(st.session_state.selected_category)
            st.session_state.feedback = None
            st.session_state.user_answer = ""
            st.rerun()
    
    # Display feedback if available
    if st.session_state.feedback:
        st.markdown("---")
        
        # Show the user's answer
        with st.expander("📝 Your Answer", expanded=False):
            st.markdown(st.session_state.user_answer)
        
        # Show feedback
        st.subheader("📋 Feedback")
        st.markdown(st.session_state.feedback)
        
        st.markdown("---")
        if st.button("Next Question →", type="primary"):
            st.session_state.current_question = get_random_question(st.session_state.selected_category)
            st.session_state.feedback = None
            st.session_state.user_answer = ""
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ML internship preparation | Powered by Gemini API")