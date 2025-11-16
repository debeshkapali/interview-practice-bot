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
    ],
    
    "Deep Learning Fundamentals": [
        {
            "id": 21,
            "question": "Explain backpropagation. How does it work?",
            "hints": ["Chain rule", "Gradient flow backward through network"],
            "key_points": ["Computes gradients using chain rule", "Propagates error backward through layers", "Essential for training neural networks"]
        },
        {
            "id": 22,
            "question": "What is a Convolutional Neural Network (CNN) and when would you use it?",
            "hints": ["Spatial data", "Image processing", "Convolutional layers"],
            "key_points": ["Designed for spatial/grid data", "Uses convolutional layers to detect patterns", "Primarily used for computer vision tasks", "Parameter sharing and translation invariance"]
        },
        {
            "id": 23,
            "question": "Explain batch normalization and why it's useful.",
            "hints": ["Normalizing layer inputs", "Training stability"],
            "key_points": ["Normalizes inputs to each layer", "Reduces internal covariate shift", "Allows higher learning rates", "Acts as regularization"]
        },
        {
            "id": 24,
            "question": "What is dropout and how does it prevent overfitting?",
            "hints": ["Randomly dropping neurons", "Ensemble effect"],
            "key_points": ["Randomly deactivates neurons during training", "Forces network to learn redundant representations", "Acts as ensemble of multiple networks", "Only used during training"]
        },
        {
            "id": 25,
            "question": "Explain transfer learning and when you would use it.",
            "hints": ["Pre-trained models", "Fine-tuning", "Small datasets"],
            "key_points": ["Using pre-trained model on new task", "Useful with limited data", "Fine-tune last layers or freeze early layers", "Leverages learned features"]
        },
        {
            "id": 26,
            "question": "What is the vanishing gradient problem and how can you address it?",
            "hints": ["Deep networks", "Sigmoid/tanh activations", "Gradient flow"],
            "key_points": ["Gradients become very small in deep networks", "Makes training early layers difficult", "Solutions: ReLU, batch norm, residual connections", "Skip connections help gradient flow"]
        },
        {
            "id": 27,
            "question": "Explain the difference between RNN and LSTM.",
            "hints": ["Sequential data", "Memory", "Long-term dependencies"],
            "key_points": ["RNN: basic recurrent architecture for sequences", "LSTM: has gates (forget, input, output)", "LSTM better at long-term dependencies", "LSTM addresses vanishing gradient in RNNs"]
        },
        {
            "id": 28,
            "question": "What are residual connections (skip connections) and why are they important?",
            "hints": ["ResNet", "Deep networks", "Gradient flow"],
            "key_points": ["Allow input to skip layers", "Enable training very deep networks", "Help gradient flow", "Prevent degradation problem"]
        },
    ],
    
    "Model Evaluation & Metrics": [
        {
            "id": 29,
            "question": "When would you use accuracy vs F1 score for model evaluation?",
            "hints": ["Class imbalance", "False positives vs false negatives"],
            "key_points": ["Accuracy: good for balanced datasets", "F1: better for imbalanced classes", "F1 balances precision and recall", "Accuracy can be misleading with imbalance"]
        },
        {
            "id": 30,
            "question": "Explain ROC curve and AUC. What do they tell you?",
            "hints": ["True positive rate vs false positive rate", "Threshold-independent"],
            "key_points": ["ROC: plots TPR vs FPR at different thresholds", "AUC: area under ROC curve", "Measures classification quality across thresholds", "Higher AUC = better model"]
        },
        {
            "id": 31,
            "question": "How do you handle class imbalance in your dataset?",
            "hints": ["Resampling", "Class weights", "Different metrics"],
            "key_points": ["Oversampling minority class or undersampling majority", "Use class weights in loss function", "SMOTE for synthetic samples", "Use appropriate metrics (F1, precision-recall)"]
        },
        {
            "id": 32,
            "question": "What is the difference between validation set and test set?",
            "hints": ["Model selection vs final evaluation", "When to use each"],
            "key_points": ["Validation: tune hyperparameters and select model", "Test: final, unbiased evaluation", "Test set should never influence training", "Validation used multiple times, test only once"]
        },
        {
            "id": 33,
            "question": "Explain mean squared error (MSE) vs mean absolute error (MAE).",
            "hints": ["Regression metrics", "Sensitivity to outliers"],
            "key_points": ["MSE: squares errors, sensitive to outliers", "MAE: absolute errors, more robust to outliers", "MSE penalizes large errors more", "Choice depends on problem requirements"]
        },
        {
            "id": 34,
            "question": "What is stratified sampling and when would you use it?",
            "hints": ["Train/test split", "Class proportions", "Imbalanced data"],
            "key_points": ["Maintains class proportions in splits", "Important for imbalanced datasets", "Ensures each split is representative", "Used in stratified k-fold cross-validation"]
        },
        {
            "id": 35,
            "question": "How do you know if your model is overfitting or underfitting?",
            "hints": ["Training vs validation performance", "Learning curves"],
            "key_points": ["Overfitting: high train accuracy, low validation accuracy", "Underfitting: low accuracy on both", "Use learning curves to diagnose", "Compare train and validation metrics"]
        },
    ],
    
    "Data Preprocessing & Feature Engineering": [
        {
            "id": 36,
            "question": "What are different ways to handle missing data?",
            "hints": ["Deletion", "Imputation", "Context matters"],
            "key_points": ["Remove rows/columns with missing data", "Impute with mean/median/mode", "Use advanced imputation (KNN, model-based)", "Consider why data is missing (MCAR, MAR, MNAR)"]
        },
        {
            "id": 37,
            "question": "Explain the difference between normalization and standardization.",
            "hints": ["Scaling features", "Range vs distribution"],
            "key_points": ["Normalization: scales to [0,1] range", "Standardization: zero mean, unit variance", "Use standardization for Gaussian-distributed data", "Normalization when you need bounded range"]
        },
        {
            "id": 38,
            "question": "When would you use one-hot encoding vs label encoding?",
            "hints": ["Categorical variables", "Nominal vs ordinal"],
            "key_points": ["One-hot: for nominal (no order) categories", "Label encoding: for ordinal (ordered) categories", "One-hot prevents model from assuming order", "Label encoding more memory efficient"]
        },
        {
            "id": 39,
            "question": "What is feature scaling and why is it important?",
            "hints": ["Different units/ranges", "Algorithm sensitivity", "Gradient descent"],
            "key_points": ["Makes features comparable in scale", "Important for distance-based algorithms", "Speeds up gradient descent", "Not needed for tree-based models"]
        },
        {
            "id": 40,
            "question": "Explain data augmentation and give examples.",
            "hints": ["Artificially increasing dataset", "Computer vision", "Preventing overfitting"],
            "key_points": ["Creates modified versions of training data", "Images: rotation, flipping, cropping, color changes", "Text: synonym replacement, back-translation", "Helps reduce overfitting and improves generalization"]
        },
        {
            "id": 41,
            "question": "What is feature selection and why is it important?",
            "hints": ["Reducing dimensionality", "Removing irrelevant features"],
            "key_points": ["Selecting most relevant features", "Reduces overfitting and training time", "Methods: filter, wrapper, embedded", "Improves model interpretability"]
        },
        {
            "id": 42,
            "question": "How would you detect and handle outliers in your data?",
            "hints": ["Statistical methods", "Visualization", "Domain knowledge"],
            "key_points": ["Detection: z-score, IQR, visualization", "Handling: remove, cap, transform, or keep", "Consider domain context", "Impact depends on algorithm used"]
        },
        {
            "id": 43,
            "question": "What is the curse of dimensionality and how does it affect ML models?",
            "hints": ["High-dimensional data", "Distance metrics", "Sparsity"],
            "key_points": ["Data becomes sparse in high dimensions", "Distance metrics become less meaningful", "Requires exponentially more data", "Dimensionality reduction helps (PCA, feature selection)"]
        },
    ],
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