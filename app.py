"""
ML/AI Interview Practice Bot
A simple app to practice interview questions and get AI feedback
"""

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
from datetime import datetime
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
if 'review_questions' not in st.session_state:
    st.session_state.review_questions = set()  # Store question IDs to review
if 'practice_mode' not in st.session_state:
    st.session_state.practice_mode = "normal"  # "normal" or "review"

def get_random_question(category="All Categories", mode="normal"):
    """Get a random question from the question bank"""
    if mode == "review":
        # Get only questions marked for review
        all_questions = get_all_questions()
        review_questions = [q for q in all_questions if q['id'] in st.session_state.review_questions]
        if category != "All Categories":
            review_questions = [q for q in review_questions if q['category'] == category]
        return random.choice(review_questions) if review_questions else None
    else:
        # Normal mode
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
    """Get AI feedback on the user's answer in journal format"""
    import time
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""You are an experienced ML/AI interviewer helping a candidate practice for internship interviews.

Question: {question['question']}

Key points that should be covered:
{', '.join(question['key_points'])}

Candidate's answer:
{user_answer}

Please provide feedback in this EXACT structured format:

## 📊 Performance Summary
Rating: [X/5]
[One sentence overall assessment]

## ✅ What You Did Well
[List 2-3 specific strengths in bullet points]

## ❌ Key Points Missed
[List the important concepts/points they didn't cover]

## 💡 Suggestions for Improvement
[Specific, actionable advice on how to improve the answer]

## 📝 Journal Entry Template
Copy this to your learning journal:

---
**Question:** {question['question']}
**Category:** {question['category']}
**Date Practiced:** {datetime.now().strftime('%Y-%m-%d')}

**What I Missed:**
[Summarize the key gaps from feedback above]

**Better Answer Outline:**
[Provide a brief outline of what a complete answer should include]

**Re-practice Date:** _____
---

Keep the feedback encouraging and specific. Remember, this is practice for an internship position."""

            response = model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a temporary error that we should retry
            if "503" in error_msg or "overloaded" in error_msg.lower() or "timeout" in error_msg.lower():
                if attempt < max_retries - 1:  # Not the last attempt
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return f"⚠️ The API is currently overloaded. Please try again in a few minutes.\n\nTechnical details: {error_msg}"
            else:
                # For other errors, fail immediately
                return f"Error getting feedback: {error_msg}"
    
    return "Unable to get feedback after multiple attempts. Please try again later."

# App UI
st.title("🤖 ML Interview Practice Bot")
st.markdown("Practice ML/AI interview questions and get instant feedback!")

# Sidebar with stats and category filter
with st.sidebar:
    st.header("📊 Your Progress")
    st.metric("Total Questions Answered", st.session_state.questions_answered)
    st.metric("Questions for Review", len(st.session_state.review_questions))
    
    st.markdown("---")
    st.subheader("Progress by Category")
    for category, count in st.session_state.category_stats.items():
        st.markdown(f"**{category}:** {count} questions")
    
    st.markdown("---")
    st.subheader("🎯 Practice Mode")
    
    # Mode selector
    mode = st.radio(
        "Select mode:",
        ["Normal Practice", "Review Mode"],
        index=0 if st.session_state.practice_mode == "normal" else 1
    )
    
    if mode == "Review Mode" and len(st.session_state.review_questions) == 0:
        st.warning("⚠️ No questions marked for review yet!")
        st.session_state.practice_mode = "normal"
    else:
        st.session_state.practice_mode = "review" if mode == "Review Mode" else "normal"
    
    # Category selector
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
    
    if st.button("🔄 Reset All Progress"):
        st.session_state.questions_answered = 0
        st.session_state.category_stats = {category: 0 for category in QUESTIONS.keys()}
        st.session_state.review_questions = set()
        st.session_state.current_question = None
        st.session_state.feedback = None
        st.session_state.user_answer = ""
        st.rerun()

# Main content area
mode_emoji = "🔄" if st.session_state.practice_mode == "review" else "📝"
mode_text = "Review Mode" if st.session_state.practice_mode == "review" else "Normal Practice"

if st.session_state.current_question is None:
    st.info(f"{mode_emoji} **{mode_text}** | Ready to practice? Click below to get a question from **{st.session_state.selected_category}**!")
    
    if st.session_state.practice_mode == "review" and len(st.session_state.review_questions) == 0:
        st.warning("You haven't marked any questions for review yet. Switch to Normal Practice to get started!")
    else:
        if st.button("Get Question", type="primary"):
            question = get_random_question(st.session_state.selected_category, st.session_state.practice_mode)
            if question:
                st.session_state.current_question = question
                st.session_state.feedback = None
                st.session_state.user_answer = ""
                st.rerun()
            else:
                st.error("No questions available for this selection!")
else:
    # Display current question
    question = st.session_state.current_question
    
    # Show if this is a review question
    is_review_question = question['id'] in st.session_state.review_questions
    
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.subheader(f"📝 {question['category']}")
    with col_header2:
        if is_review_question:
            st.markdown("🔄 *Review*")
    
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
    
    col1, col2, col3 = st.columns(3)
    
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
            st.session_state.current_question = get_random_question(st.session_state.selected_category, st.session_state.practice_mode)
            st.session_state.feedback = None
            st.session_state.user_answer = ""
            st.rerun()
    
    with col3:
        # Toggle review status
        if is_review_question:
            if st.button("✓ Remove Review"):
                st.session_state.review_questions.remove(question['id'])
                st.rerun()
        else:
            if st.button("🔖 Mark Review"):
                st.session_state.review_questions.add(question['id'])
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
        
        # Quick action buttons after feedback
        st.markdown("---")
        col_next1, col_next2, col_next3 = st.columns(3)
        
        with col_next1:
            if st.button("Next Question →", type="primary"):
                st.session_state.current_question = get_random_question(st.session_state.selected_category, st.session_state.practice_mode)
                st.session_state.feedback = None
                st.session_state.user_answer = ""
                st.rerun()
        
        with col_next2:
            if not is_review_question:
                if st.button("🔖 Mark for Review"):
                    st.session_state.review_questions.add(question['id'])
                    st.success("Added to review list!")
                    st.rerun()
        
        with col_next3:
            if is_review_question:
                if st.button("✓ Mastered This!"):
                    st.session_state.review_questions.remove(question['id'])
                    st.success("Removed from review list!")
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for ML internship preparation | Powered by Gemini API")