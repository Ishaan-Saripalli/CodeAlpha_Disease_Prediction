import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data_path = r"C:\Users\DELL\Desktop\disease_prediction\cardio_train.csv"
data = pd.read_csv(data_path, sep=';')
print("Dataset Head:\n", data.head())
print("Data Info:\n", data.info())
X = data.drop(columns=['cardio', 'id'])  
y = data['cardio'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:\n", feature_importances)


'''
import joblib
model_path = r"C:\\Users\\DELL\\Desktop\\disease_prediction\\cardio_model.pkl"
joblib.dump(model, model_path)
print("\nModel saved successfully!")
'''
