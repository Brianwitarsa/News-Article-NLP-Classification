# News Article Routing with Classical ML and CNN (AG News)

This project develops an automated system to **separate Sports news articles from non-Sports content** (Business, World, and Sci/Tech) within a newly integrated mixed news collection. Using the AG News dataset, the analysis combines classical machine learning and deep learning approaches to determine which modeling strategy produces the most reliable routing decisions for editorial workflows.

## Project Overview
- Exploratory Data Analysis (EDA) of class balance and text characteristics  
- Vocabulary and keyword overlap analysis between article groups  
- TF–IDF feature engineering with dimensionality reduction (Truncated SVD)  
- Classical baseline modeling using Random Forest  
- Deep learning modeling using a 1D Convolutional Neural Network (CNN)  
- Model comparison using accuracy, precision, recall, and confusion matrices  

## Dataset
The dataset includes:
- 120,000 training articles and 7,600 test articles  
- News categories: Sports, Business, World, and Sci/Tech  
- Article titles and descriptions combined into a single modeling text field  
- Binary routing target:
  - Sports  
  - Non-Sports (Business + World + Sci/Tech)

## Exploratory Data Analysis (EDA)
Key findings:
- The dataset is balanced across the original four categories, enabling a stable binary split.  
- Article length features (title/description length) show **near-zero correlation** with the routing label.  
- Vocabulary analysis shows **strong linguistic separation** between Sports and non-Sports content: only **4 of the top 50 words overlapped**, indicating that keyword-based signals are highly informative for classification.

## Feature Engineering
Text preprocessing included:
- TF–IDF vectorization with unigrams and bigrams  
- Stop-word removal and vocabulary filtering (min_df / max_df thresholds)  
- Limiting to 5,000 features for computational efficiency  
- Dimensionality reduction using **Truncated SVD (300 components)** to create dense feature representations for classical models  

## Classical Baseline Model
A **Random Forest classifier** was trained on the reduced TF–IDF features:
- Test Accuracy: **96.28%**  
- Provided a strong baseline with minimal tuning  
- Slight overfitting observed due to ensemble complexity  

## Deep Learning Model (CNN)
A **1D Convolutional Neural Network** was trained on tokenized article sequences:
- Embedding layer followed by Conv1D, global max pooling, and dense layers  
- Early stopping and dropout used to control overfitting  
- Test Accuracy: **98.37%**  
- Precision:
  - Sports: **~96.1%**
  - Non-Sports: **~99.1%**

The CNN captured phrase-level context and sequential patterns, leading to improved routing reliability compared to the classical baseline.

## Technologies Used
- Python  
- TensorFlow / Keras  
- scikit-learn  
- pandas / NumPy  
- matplotlib / seaborn  

## Results Summary
- Early vocabulary analysis confirmed that **Sports vs non-Sports** is a well-separated classification task.  
- The **Random Forest baseline** delivered strong performance using compressed TF–IDF features.  
- The **CNN model achieved the best results**, improving precision and overall routing reliability by learning contextual word patterns directly from text sequences.  


Links: 
- [Code](https://github.com/Brianwitarsa/New-Article_Classification/blob/main/Brian_Witarsa_AML_Final.ipynb)
