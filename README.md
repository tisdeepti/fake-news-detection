# Fake News Detection using Machine Learning

## Description
Fake News Detection is a machine learning project that identifies and classifies news articles as either **Fake News** or **True News**. The goal is to combat the spread of misinformation by leveraging machine learning techniques.

This project uses datasets containing fake and true news articles, processes the data, and trains models to achieve high accuracy in detecting fake news. The models use text vectorization techniques like **TF-IDF** and algorithms such as Logistic Regression for classification.

---

## Features
- Preprocessing text data by cleaning and tokenizing.
- TF-IDF Vectorization for feature extraction.
- Logistic Regression for classification with high accuracy.
- Manual testing with real-time input to detect fake news.
- Modular code structure for flexibility and easy updates.

---

## Dataset
- **Fake News Dataset**: Contains a collection of fake news articles.
- **True News Dataset**: Contains verified news articles.
- Downloaded from [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

---

## Libraries Used
- `pandas`
- `numpy`
- `re` (Regex)
- `string`
- `scikit-learn`

---

## Prerequisites
Before running the project, ensure the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`

---

## File Structure
```
.
├── fake.csv
├── true.csv
├── hybrid_fake_news.py
├── README.md
```

---

## Project Workflow
1. **Data Preprocessing**:
   - Text is cleaned by removing punctuation, special characters, and URLs.
   - TF-IDF is applied for feature extraction.
2. **Model Training**:
   - Logistic Regression is used to classify news articles.
   - The model is evaluated using metrics like precision, recall, and F1-score.
3. **Manual Testing**:
   - Users can input news articles to classify them as Fake or True.
4. **Example Outputs**:
   - **Input**:  
     `"Breaking News: The moon has disappeared from the night sky!"`  
   - **Output**:  
     `Logistic Regression: Fake News`

---

## Future Enhancements
- Incorporate deep learning models like LSTMs for better accuracy.
- Add a web interface for easier usability.
- Expand the dataset for improved generalization.

---

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## License
This project is licensed under the MIT License.
```

