from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import pathlib
import joblib
import encode
import training


# Directory containing the dataset organized by emotion
file_path_list = list(pathlib.Path("datas_ck+mod").glob('*/*.png')) + list(pathlib.Path("datas_ck+mod").glob('*/*.jpg'))
file_label_list = [str(file_path).split('/')[1] for file_path  in file_path_list]

print(f"Number of files: {len(file_path_list)}")
print(list(pathlib.Path("datas_ck+mod").glob('*/*.jpg')))

datasetcsv_path = "features.csv"
features = None

# Save features and labels to a CSV file for later use
if not pathlib.Path(datasetcsv_path).exists():
    features = encode.prepare_datas(file_path_list, file_label_list)
    features.to_csv(datasetcsv_path, index=False)
    print(f"Data extraction and pose estimation complete. Features saved to '{datasetcsv_path}'.")
else:
    features = pd.read_csv(datasetcsv_path)
    print(f"Data Load from {datasetcsv_path}.")

X_train, X_test, y_train, y_test = encode.data_prep_split(features)

y_all = list(set(features['emotion']))
print(y_all)

#param_grid = {
    #    'n_estimators': [100, 200, 300],
    #    'max_depth': [None, 10, 20],
    #    'min_samples_split': [2, 5, 10],
    #    'min_samples_leaf': [1, 2, 4]
    #}

param_grid = {
    'n_neighbors': [10, 50, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
}

#model = RandomForestClassifier(random_state=42)
model = training.train_model(KNeighborsClassifier(), param_grid
                            , X_train, X_test, y_train, y_test)

# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
joblib.dump(model, 'model.pkl')

# Save the trained model to a file
print("Model saved as 'emotion_classifier.pkl'.")


cm = confusion_matrix(y_test, y_pred, labels=y_all)
report = classification_report(y_test, y_pred)
with open('classification_report.txt', 'w') as f:
    f.write(report)
    f.write(str(y_all))
    f.write(str(cm))
print("Classification report saved as 'classification_report.txt'.")