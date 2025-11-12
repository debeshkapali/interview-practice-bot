"""
Question bank for ML/AI interview practice
"""

QUESTIONS = {
    "ML Fundamentals": [
        {
            "id": 1,
            "question": "Explain the bias-variance tradeoff in machine learning.",
            "hints": ["Think about underfitting and overfitting", "Consider model complexity"],
            "key_points": ["Bias: error from overly simple assumptions", "Variance: error from sensitivity to training data", "Tradeoff: need to balance both"]
        },
        {
            "id": 2,
            "question": "What is overfitting and how can you prevent it?",
            "hints": ["Model performs well on training but poorly on test data", "Think about regularization techniques"],
            "key_points": ["Overfitting: model learns noise in training data", "Prevention: regularization, dropout, early stopping, more data, simpler model"]
        },
        {
            "id": 3,
            "question": "Explain the difference between supervised and unsupervised learning.",
            "hints": ["Think about labeled vs unlabeled data"],
            "key_points": ["Supervised: learns from labeled data", "Unsupervised: finds patterns in unlabeled data", "Examples of each"]
        },
        {
            "id": 4,
            "question": "What is cross-validation and why is it important?",
            "hints": ["Think about model evaluation", "K-fold"],
            "key_points": ["Splitting data into folds", "Reduces overfitting to validation set", "More reliable performance estimate"]
        },
        {
            "id": 5,
            "question": "Explain the difference between classification and regression.",
            "hints": ["Think about output types"],
            "key_points": ["Classification: discrete outputs/categories", "Regression: continuous outputs", "Different metrics for each"]
        },
        {
            "id": 6,
            "question": "What is regularization and why do we use it?",
            "hints": ["L1 and L2 regularization", "Prevents overfitting"],
            "key_points": ["Adds penalty to loss function", "L1: Lasso, L2: Ridge", "Reduces model complexity"]
        },
        {
            "id": 7,
            "question": "Explain precision and recall. When would you prioritize one over the other?",
            "hints": ["Think about false positives vs false negatives"],
            "key_points": ["Precision: correctness of positive predictions", "Recall: coverage of actual positives", "Use cases for each"]
        },
        {
            "id": 8,
            "question": "What is a confusion matrix and what does it tell you?",
            "hints": ["TP, TN, FP, FN"],
            "key_points": ["Shows model's predictions vs actual labels", "Helps calculate precision, recall, accuracy", "Identifies types of errors"]
        },
    ],
    
    "Math for ML": [
        {
            "id": 9,
            "question": "Explain gradient descent. How does it work?",
            "hints": ["Think about optimization", "Learning rate"],
            "key_points": ["Iterative optimization algorithm", "Moves in direction of steepest descent", "Updates weights using gradients"]
        },
        {
            "id": 10,
            "question": "What is a loss function? Give examples for classification and regression.",
            "hints": ["Measures model error", "Different for different tasks"],
            "key_points": ["Quantifies prediction error", "Classification: cross-entropy", "Regression: MSE, MAE"]
        },
        {
            "id": 11,
            "question": "Explain the concept of a derivative and why it's important in ML.",
            "hints": ["Rate of change", "Backpropagation"],
            "key_points": ["Measures rate of change", "Used to compute gradients", "Essential for optimization"]
        },
        {
            "id": 12,
            "question": "What is the purpose of activation functions in neural networks?",
            "hints": ["Non-linearity", "ReLU, sigmoid, tanh"],
            "key_points": ["Introduce non-linearity", "Allow learning complex patterns", "Different types for different purposes"]
        },
        {
            "id": 13,
            "question": "Explain what a matrix multiplication is and why it's fundamental to neural networks.",
            "hints": ["Linear transformations", "Weights and inputs"],
            "key_points": ["Linear transformation operation", "Combines weights with inputs", "Core of neural network forward pass"]
        },
        {
            "id": 14,
            "question": "What is the difference between a parameter and a hyperparameter?",
            "hints": ["Learned vs set by you"],
            "key_points": ["Parameters: learned during training (weights, biases)", "Hyperparameters: set before training (learning rate, batch size)", "Examples of each"]
        },
    ],
    
    "PyTorch Basics": [
        {
            "id": 15,
            "question": "What is a PyTorch tensor and how is it different from a NumPy array?",
            "hints": ["GPU acceleration", "Automatic differentiation"],
            "key_points": ["Multi-dimensional array", "Can run on GPU", "Supports autograd for backprop"]
        },
        {
            "id": 16,
            "question": "Explain what autograd is in PyTorch.",
            "hints": ["Automatic differentiation", "Gradients"],
            "key_points": ["Automatically computes gradients", "Builds computational graph", "Essential for backpropagation"]
        },
        {
            "id": 17,
            "question": "What is the purpose of torch.nn.Module in PyTorch?",
            "hints": ["Base class for models", "Forward method"],
            "key_points": ["Base class for all neural networks", "Must implement forward() method", "Manages parameters automatically"]
        },
        {
            "id": 18,
            "question": "Explain the basic structure of a training loop in PyTorch.",
            "hints": ["Forward pass, loss, backward, optimizer step"],
            "key_points": ["Forward pass: compute predictions", "Calculate loss", "Backward pass: compute gradients", "Optimizer step: update weights"]
        },
        {
            "id": 19,
            "question": "What is the difference between model.eval() and model.train() in PyTorch?",
            "hints": ["Training vs inference mode", "Dropout, batch norm"],
            "key_points": ["train(): enables dropout, batch norm updates", "eval(): disables dropout, uses running stats", "Important for correct model behavior"]
        },
        {
            "id": 20,
            "question": "What does loss.backward() do in PyTorch?",
            "hints": ["Backpropagation", "Computes gradients"],
            "key_points": ["Computes gradients via backpropagation", "Stores gradients in .grad attribute", "Must be called before optimizer.step()"]
        },
    ]
}

def get_all_questions():
    """Returns a flat list of all questions with their categories"""
    all_questions = []
    for category, questions in QUESTIONS.items():
        for q in questions:
            q_copy = q.copy()
            q_copy['category'] = category
            all_questions.append(q_copy)
    return all_questions

def get_questions_by_category(category):
    """Returns questions for a specific category"""
    return QUESTIONS.get(category, [])