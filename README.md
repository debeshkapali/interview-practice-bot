# 🤖 ML Interview Practice Bot

> A personal project built to overcome interview anxiety and prepare for ML/AI internships through structured practice and AI-powered feedback.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 About This Project

As someone transitioning from digital marketing/SEO to ML/AI, I found traditional mock interviews intimidating due to my introverted nature and anxiety. This tool allows me to practice interview questions at my own pace, receive constructive feedback, and track my progress—all without the pressure of real-time human interaction.

### The Problem
- Interview anxiety makes traditional mock interviews stressful
- Difficult to get consistent, objective feedback on technical answers
- Hard to identify and track weak areas systematically
- Need flexible practice that fits around learning schedule

### The Solution
A simple, focused web app that:
- Presents curated ML/AI interview questions
- Provides AI-powered feedback on answers
- Tracks progress by topic area
- Allows category-specific practice

## ✨ Features

- **20 Curated Questions** across three key areas:
  - ML Fundamentals (8 questions)
  - Math for ML (6 questions)
  - PyTorch Basics (6 questions)

- **Category-Based Practice**: Choose to practice all categories or focus on specific areas

- **AI-Powered Feedback**: Get constructive, detailed feedback on your answers including:
  - What you did well
  - Key points you missed
  - Suggestions for improvement
  - Rating out of 5

- **Progress Tracking**: Monitor your practice with:
  - Total questions answered
  - Category-wise breakdown
  - Ability to reset and start fresh

- **Hints System**: Get gentle nudges if you're stuck without seeing the full answer

- **Answer Review**: Compare your written answer with the feedback received

## 🛠️ Tech Stack

- **Frontend**: Streamlit - Chose for rapid prototyping and clean UI
- **AI Model**: Google Gemini 2.5 Flash - Free tier with excellent feedback quality
- **Language**: Python 3.9+
- **Environment Management**: python-dotenv for secure API key handling

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher (or Anaconda/Miniconda)
- Google account (for Gemini API key)

### Installation

1. **Clone the repository**
```bash
git clone git@github.com:debeshkapali/interview-practice-bot.git
cd interview-practice-bot
```

2. **Set up your environment**

You can use either **pip** (with venv) or **conda**:

#### Option A: Using pip and venv
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda (Recommended for ML users)
```bash
# Create conda environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate streamlit-genai-env
```

4. **Get your Gemini API key**
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Sign in with your Google account
- Click "Create API Key"
- Copy the generated key

5. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Open .env and add your API key (no spaces around =):
GEMINI_API_KEY=your_api_key_here
```

6. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📚 Usage

1. **Select a category** from the sidebar (or keep "All Categories" for variety)
2. Click **"Get Question"** to receive a random question
3. **Read carefully** and use hints if needed
4. **Type your answer** in the text area
5. Click **"Submit Answer"** to receive AI feedback
6. **Review feedback** and your original answer
7. Click **"Next Question"** to continue practicing

### Tips for Best Results
- Take your time - quality matters more than speed
- Try to answer without looking at hints first
- Be thorough but concise in your explanations
- Review feedback carefully to improve future answers
- Practice regularly (3-5 questions per day recommended)

## 📂 Project Structure

```
interview-practice-bot/
├── app.py              # Main Streamlit application
├── questions.py        # Question bank with hints and key points
├── check-models.py     # Checks and lists available models
├── requirements.txt    # Python dependencies
├── environment.yml     # Create conda environment
├── .env                # API key (not committed to git)
├── .env.example        # Template for environment variables
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🎯 Future Enhancements

Ideas for Phase 2 (based on actual usage feedback):

- [ ] **Save practice sessions** - Review past answers and track improvement
- [ ] **Weak area identification** - Flag and prioritize topics you struggle with
- [ ] **Difficulty levels** - Tag questions as Easy/Medium/Hard
- [ ] **Timed practice mode** - Simulate real interview time pressure
- [ ] **Custom questions** - Add your own questions from real interviews
- [ ] **Voice input** - Record answers instead of typing
- [ ] **Export functionality** - Download practice history as PDF/CSV
- [ ] **Spaced repetition** - Automatically review difficult questions

## 🧠 What I Learned

Building this project taught me:
- Working with LLM APIs (Google Gemini)
- Building interactive web apps with Streamlit
- Environment variable management and security best practices
- Debugging API integration issues
- Iterative development and MVP thinking
- Creating meaningful portfolio projects that solve real problems

### Challenges Overcome
1. **API Key Configuration**: Learned about proper environment variable handling and common pitfalls (spaces around `=`, wrong file locations)
2. **Model Availability**: Navigated deprecation of `gemini-pro` and learned to check available models programmatically
3. **State Management**: Understood Streamlit's session state for maintaining app data across reruns

## 👨‍💻 About Me

I'm currently transitioning from digital marketing/SEO (3 years) to ML/AI. This project is part of my journey to build practical skills and portfolio projects while learning ML fundamentals.

**Background:**
- 3 years in digital marketing/SEO
- Currently learning: PyTorch, TensorFlow, ML fundamentals
- Previous experience: Deployed a Streamlit app
- Goals: Land an ML/AI internship and contribute to the field

**Why I built this:**
As an introvert, traditional mock interviews felt overwhelming. I needed a judgment-free space to practice and improve at my own pace.

## 🤝 Contributing

This is primarily a personal learning project, but I'm open to suggestions! Feel free to:
- Open issues for bugs or feature ideas
- Share your experience if you use this for your own prep
- Suggest additional interview questions

## 📝 License

MIT License - Feel free to use this for your own interview preparation!

## 🙏 Acknowledgments

- Thanks to the ML community for sharing interview experiences
- Google for providing free Gemini API access
- Streamlit for making web app development accessible

---

**Built with ❤️ as part of my ML/AI career transition journey**

*If this project helped you, consider giving it a ⭐ on GitHub!*

---

## 📧 Connect With Me

- GitHub: [@debeshkapali](https://github.com/debeshkapali)
- LinkedIn: [debesh-kapali](https://www.linkedin.com/in/debesh-kapali/)

*Currently seeking ML/AI internship opportunities!*