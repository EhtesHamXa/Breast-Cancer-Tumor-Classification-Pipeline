import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer

# 1. Load the raw dictionary-like object from Scikit-Learn
raw_data = load_breast_cancer()
# 2. Convert it into a clean Pandas DataFrame
# We give it the raw numbers (.data) and label the columns (.feature_names)
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
# 3. Add the 'Target' column (The answers)
df['Target'] = raw_data.target
# --- IN-DEPTH EDA (Exploratory Data Analysis) ---
# Look at the first 5 rows to understand the math we are working with
print("--- Data Snapshot ---")
print(df.head())
# Look for Class Imbalance!
print("\n--- Target Distribution ---")
# value_counts() counts how many 0s and 1s we have. 
# normalize=True mathematically turns it into a percentage!
distribution = df['Target'].value_counts(normalize=True) * 100
print(distribution)
# Let's visualize the Imbalance
sns.countplot(x='Target', data=df)
plt.title("Class Imbalance Check (0 = Malignant, 1 = Benign)")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- STEP 2: PREPROCESSING ---
# 1. Separate Features (X) and Target (y)
X = df.drop('Target', axis=1) # Drop the answer column to get our features
y = df['Target']              # Save the answers in y
# 2. Train/Test Split (CRITICAL: Do this BEFORE scaling!)
# stratify=y mathematically ensures the 37/63 ratio is preserved in both the practice and final exams!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 3. Scale the Data!
scaler = StandardScaler()
# .fit_transform() calculates the Mean of the training data AND applies the scaling
X_train_scaled = scaler.fit_transform(X_train)
# .transform() ONLY applies the scaling using the mathematical rules it learned from the training data!
# (This prevents Data Leakage from the test set!)
X_test_scaled = scaler.transform(X_test)
print("\n--- Preprocessing Complete ---")
print(f"Original Tumor Area Example: {X_train.iloc[0]['mean area']:.2f}")
print(f"Scaled Tumor Area Example:   {X_train_scaled[0][3]:.2f}") # Column index 3 is 'mean area'


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# --- STEP 3: MODEL TRAINING ---
# 1. Initialize the Random Forest
# class_weight='balanced' mathematically forces the AI to pay extra attention to the minority class (Malignant)!
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# 2. Train the model on the SCALED training data!
model.fit(X_train_scaled, y_train)
# --- STEP 4: MODEL EVALUATION ---
# 1. Give the Final Exam (Predict on the SCALED test data)
predictions = model.predict(X_test_scaled)
# 2. Grade the Model
print("\n--- Model Evaluation ---")
print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))
print("\n--- Advanced Classification Report ---")
# This prints Precision, Recall, and F1-Score for BOTH classes!
print(classification_report(y_test, predictions, target_names=['Malignant (Cancer)', 'Benign (Safe)']))