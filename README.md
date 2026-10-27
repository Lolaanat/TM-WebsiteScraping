# Text Mining — Game Review Rating Prediction

## Project Overview

This project focuses on predicting game review ratings using text mining and machine learning techniques. The analysis uses reviews collected from the Google Play Store for the Roblox game, with the objective of exploring review patterns, preprocessing text data, comparing text representation methods, and evaluating different prediction models.

The project was developed as part of the **Text Mining Midterm Examination** at BINUS University.

## Objectives

- Collect 2,000 game reviews with different ratings.
- Explore dominant words for each rating and identify frequently occurring non-standard words.
- Preprocess text while preserving word meaning and grammatical categories as much as possible.
- Compare TF-IDF and Word2Vec as text representation methods.
- Build and tune machine learning and deep learning models to predict review ratings.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Handle class imbalance using SMOTE and evaluate the resulting model performance.

## Dataset

The dataset was collected from the Google Play Store using the `google-play-scraper` library.

| Attribute | Description |
|---|---|
| Application | Roblox |
| Application ID | `com.roblox.client` |
| Number of reviews | 2,000 |
| Target variable | Rating (1–5) |
| Original review language | Indonesian |
| Analysis language | English |
| Output file | `roblox_review.csv` |

The original reviews were collected in Indonesian and translated into English using `deep-translator` to support English-language text analysis.

The dataset contains the following columns:

- `User Name`: Reviewer username.
- `Rating`: Rating given by the reviewer, from 1 to 5.
- `Date`: Review submission date.
- `Ulasan`: Original review text.
- `Review`: English-translated review.
- `Likes`: Number of thumbs-up received by the review.

After preprocessing, an additional column named `Processed_Review` was created.

## Methodology

### 1. Data Collection

A total of 2,000 Roblox reviews were collected from the Google Play Store using `google-play-scraper`. The reviews were retrieved in Indonesian and translated into English using `GoogleTranslator` from the `deep-translator` library.

The collected data was stored in a Pandas DataFrame and exported to `roblox_review.csv`.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand the characteristics of the collected reviews.

The analysis included:

- Visualizing the most frequently occurring words for each rating from 1 to 5 using bar charts.
- Identifying frequently occurring non-standard English words using `pyspellchecker`.
- Examining word frequency patterns to better understand the review data.

The analysis identified frequently occurring terms such as `roblox`, `dont`, `im`, `robux`, and `thats`.

### 3. Text Preprocessing

Text preprocessing was performed to clean and normalize the review data before modeling.

The steps included:

1. Converting text to lowercase.
2. Removing numbers and special characters.
3. Tokenizing text into individual words using NLTK.
4. Removing English stopwords.
5. Removing tokens with fewer than three characters.
6. Identifying parts of speech, including nouns, verbs, adjectives, and adverbs.
7. Applying lemmatization using `WordNetLemmatizer` while considering the identified part of speech.

The resulting processed text was stored in the `Processed_Review` column.

### 4. Text Representation

Two text representation methods were implemented and compared.

#### TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) represents words according to their importance within a document relative to the collection of documents.

Configuration:
- Maximum features: 2,000.
- Actual feature dimensions: 1,869.
- Output shape: `(2000, 1869)`.

#### Word2Vec

Word2Vec learns word vector representations based on surrounding words using a neural network approach.

Configuration:
- Vector size: 100.
- Context window: 5.
- Minimum word count: 1.
- Training method: Skip-Gram (`sg=1`).
- Workers: 4.

The word vectors were averaged to create a single 100-dimensional representation for each review.

Output shape: `(2000, 100)`.

A preliminary comparison using Random Forest produced the following results:

| Text Representation | Accuracy |
|---|---:|
| TF-IDF | 67.50% |
| Word2Vec | 61.25% |

TF-IDF achieved higher accuracy than Word2Vec in this preliminary comparison.

### 5. Machine Learning Models

Two machine learning algorithms were implemented using TF-IDF features.

#### Random Forest

Random Forest combines multiple decision trees to predict the review rating.

Hyperparameter tuning was performed using `GridSearchCV` with three-fold cross-validation.

Parameters tested:
- `n_estimators`: 50, 100.
- `max_depth`: `None`, 10.

Best parameters:
- `n_estimators`: 100.
- `max_depth`: `None`.

#### Support Vector Machine (SVM)

Support Vector Machine classifies reviews by identifying decision boundaries between rating classes.

Hyperparameter tuning was performed using `GridSearchCV` with three-fold cross-validation.

Parameters tested:
- `C`: 0.1, 1, 10.
- `kernel`: `linear`, `rbf`.

Best parameters reported by the notebook:
- `C`: 1.
- `kernel`: `linear`.

### 6. Deep Learning Model

A feed-forward neural network (Multilayer Perceptron / MLP) was implemented using Word2Vec features.

The model architecture consisted of:
- Input layer based on 100-dimensional Word2Vec vectors.
- First hidden layer with a configurable number of neurons and ReLU activation.
- Dropout layer with a rate of 0.2.
- Second hidden layer with 32 neurons and ReLU activation.
- Output layer with five neurons and softmax activation.

The model was trained using categorical cross-entropy loss.

Hyperparameter configurations tested:
- Optimizer: `adam`, `rmsprop`.
- Number of neurons in the first hidden layer: 64, 128.

Training configuration:
- Epochs: 20.
- Batch size: 32.
- Validation split: 10%.

The best validation accuracy reported was 62.50%.

### 7. Model Evaluation

The models were evaluated using accuracy, weighted precision, weighted recall, and weighted F1-score.

The notebook reported the following results:

| Text Representation | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|
| TF-IDF | Random Forest | 69.75% | 60.95% | 69.75% | 62.95% |
| TF-IDF | SVM | 67.50% | 54.17% | 67.50% | 60.10% |
| Word2Vec | Deep Learning (MLP) | 62.00% | 47.39% | 62.00% | 52.63% |

Based on these results, Random Forest with TF-IDF achieved the highest accuracy and weighted F1-score among the three models.

The SVM classification report also showed that the model did not successfully predict some minority rating classes, highlighting the effect of class imbalance on model performance.

### 8. Handling Class Imbalance with SMOTE

The original rating distribution was imbalanced:

| Rating | Number of Reviews |
|---|---:|
| 1 | 455 |
| 2 | 87 |
| 3 | 126 |
| 4 | 189 |
| 5 | 1,143 |

SMOTE (Synthetic Minority Over-sampling Technique) was applied to the TF-IDF feature matrix to increase the representation of minority classes.

After resampling, each rating class contained 1,143 samples.

An SVM model with `C=1` and `kernel='rbf'` was then trained and evaluated on the resampled dataset.

The reported classification results were:

| Rating | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| 1 | 0.89 | 0.84 | 0.86 |
| 2 | 0.97 | 0.99 | 0.98 |
| 3 | 0.97 | 0.95 | 0.96 |
| 4 | 0.93 | 0.89 | 0.91 |
| 5 | 0.80 | 0.88 | 0.84 |
| **Overall** | **0.91** | **0.91** | **0.91** |

The overall accuracy after SMOTE was **91%**.

These results suggest that balancing the rating classes improved the model's ability to classify the different ratings on the resampled evaluation data.

**Evaluation note:** The notebook applies SMOTE before splitting the resampled data into training and test sets. Therefore, the 91% result should be interpreted cautiously because this procedure can introduce data leakage and overestimate generalization performance. A more reliable evaluation would split the original data first and apply SMOTE only to the training set.

## Conclusion

Based on the experiments conducted:

1. **TF-IDF outperformed Word2Vec** in the preliminary Random Forest comparison, achieving 67.50% versus 61.25% accuracy.
2. **Random Forest with TF-IDF achieved the best baseline performance**, with 69.75% accuracy and a weighted F1-score of 62.95%.
3. The deep learning MLP model using Word2Vec achieved a validation accuracy of 62.50% and a reported test accuracy of 62.00%.
4. The original dataset had a substantial class imbalance, particularly for ratings 2, 3, and 4.
5. After applying SMOTE, the SVM model achieved 91% accuracy on the resampled evaluation data. However, this result requires further validation using a leakage-free train-test procedure.

Overall, the experiments demonstrate the importance of text representation, model selection, hyperparameter tuning, and class imbalance handling in review rating prediction.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Google Play Scraper
- Deep Translator
- NLTK
- PySpellChecker
- Gensim / Word2Vec
- Scikit-learn
- TensorFlow / Keras
- imbalanced-learn / SMOTE

## Project Files

- `UTS_TM.ipynb` — Notebook containing data collection, exploratory analysis, preprocessing, text representation, model training, evaluation, and imbalance handling.
- `roblox_review.csv` — Collected review dataset generated by the notebook.

## How to Run

1. Open `UTS_TM.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required libraries using the installation commands provided in the notebook.
3. Run the notebook cells in order.
4. Ensure an internet connection is available for review scraping, translation, and downloading NLTK resources.
5. The notebook generates `roblox_review.csv` during the data collection process.

### Author
**Leora Natania**

Text Mining — Midterm Examination
