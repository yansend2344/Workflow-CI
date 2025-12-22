import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

INPUT_PATH = "heartdisease_preprocessing.csv"

def main():
    print("--- Memulai Workflow MLProject ---")

    # Load Data
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} tidak ditemukan!")
        return
        
    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=['num'])
    y = df['num']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Docker_CI_Run"):
        
        model = RandomForestClassifier(
            n_estimators=200,  
            max_depth=10,    
            random_state=42    
        )
        
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Training Selesai.")
        print(f"Akurasi Model: {acc:.4f}") 

if __name__ == "__main__":
    main()