import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import argparse
import joblib  # For saving the model
from joblib import dump, load

# 📂 Charger le dataset
df = pd.read_csv("final_dataset.csv")

# 🔎 Vérifier la distribution des classes
print("📌 Distribution des classes :\n", df["Abnormality class"].value_counts())

# 🎯 Définir X (features) et y (target)
print("📌 Colonnes disponibles :", df.columns)
X = df.drop(columns=["Abnormality class", "Experiment"], errors="ignore")
y = df["Abnormality class"]

# 🎭 Encodage des labels (y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 🔄 Convert non-numeric columns to numeric
non_numeric_columns = X.select_dtypes(include=['object']).columns
for col in non_numeric_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 🚨 Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🎲 Séparer en train & test (80% train, 20% test) avec stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔬 Normalisation des features (important pour SVM et KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Vérification
print(f"✔️ Taille du jeu de train : {X_train.shape}")
print(f"✔️ Taille du jeu de test : {X_test.shape}")

# 📌 Initialisation des modèles
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# 🔥 Entraînement et évaluation des modèles
accuracies = {}
for name, model in models.items():
    print(f"\n🔹 Entraînement du modèle {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 🎯 Stocker l'accuracy
    accuracies[name] = accuracy_score(y_test, y_pred)
    
    # 📊 Affichage des résultats
    print(f"📌 Modèle : {name}")
    print(f"🔹 Accuracy : {accuracies[name]:.4f}")
    print(f"🔹 Rapport de classification : \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")
    print("-" * 50)

# 📊 Tracer les résultats
plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel("Modèles")
plt.ylabel("Accuracy")
plt.title("Performance des Modèles de Machine Learning")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.show()


model = KNeighborsClassifier()
#model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)


# 10. Hacer predicciones con el conjunto de validación
# Make predictions on the validation set
predictions = model.predict(X_test)
predictions = predictions.round().astype(int)

# Save the model
dump(model, 'model_KNN.joblib') 

'''
# Load the model later
loaded_model = load('model_KNN.joblib')


sample_features = X_test[10]
sample_features = np.array(sample_features).reshape(1, -1)

# 3. Make prediction
prediction = loaded_model.predict(sample_features)

# If your model does probabilities:
# probabilities = model.predict_proba(sample_features)

print(f"Predicted class: {prediction[0]}")'''