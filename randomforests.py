import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Load and Prepare Data
# Assuming the dataset is already loaded into the variable `data`
file_path = '/mnt/data/bbc-news-data (1).csv'
data = pd.read_csv(file_path, delimiter='\t')
data['combined_text'] = data['title'] + " " + data['content']

# Step 2: Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['combined_text'])
y = data['category']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = rf_classifier.predict(X_test)

# Output Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report_text)

# Step 6: Data Visualization
# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    rf_classifier, X_test, y_test, cmap=plt.cm.Blues, xticks_rotation='vertical'
)
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()
