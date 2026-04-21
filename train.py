import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from model import MLP

# Funcion para evaluar los distintos umbrales
def evaluate_thresholds(y_true, probs, thresholds):
    results = []

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm
        })

    return results

# Función para imprimir los resultados de las metricas
def print_threshold_results(results):
    print("\nThreshold comparison:")
    for result in results:
        print("\n" + "-" * 50)
        print(f"Threshold: {result['threshold']:.2f}")
        print(f"Accuracy:  {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall:    {result['recall']:.4f}")
        print(f"F1-score:  {result['f1_score']:.4f}")
        print("Confusion matrix:")
        print(result["confusion_matrix"])


def main():
    # Reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)

    # Cargamos los datos
    df = pd.read_csv("server_sensor_data.csv")

    X = df.drop("failure", axis=1).values
    y = df["failure"].values

    # Hacemos un split para test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Hacemos un split para validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # Escalamos los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convertimos a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Carga de datos para entrenamiento
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Pesos para la clase positiva y negativa
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()

    pos_weight_value = n_negative / n_positive
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    print(f"Positive class weight: {pos_weight.item():.4f}")

    # Modelo, función de perdida y optimizador
    model = MLP(input_dim=X_train_tensor.shape[1])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Definimos early stopping
    epochs = 100
    patience = 10
    min_delta = 1e-4

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    # Entrenamos el modelo
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Evaluamos en el conjunto de validación
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

        # aAplicamos early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Cargamos el mejor modelo
    model.load_state_dict(best_model_state)
    print(f"Best validation loss: {best_val_loss:.4f}")

    #Predecimos con el mejor modelo
    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    y_test_np = y_test_tensor.cpu().numpy().flatten().astype(int)

    # Evaluamos distintos umbrales
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    results = evaluate_thresholds(y_test_np, probs, thresholds)

    print_threshold_results(results)

    # Seleccionamos el mejor umbral basado en F1-score
    best_result = max(results, key=lambda x: x["f1_score"])

    print("\n" + "=" * 50)
    print("Best threshold based on F1-score")
    print("=" * 50)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Accuracy:  {best_result['accuracy']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-score:  {best_result['f1_score']:.4f}")
    print("Confusion matrix:")
    print(best_result["confusion_matrix"])

    # Guardamos el modelo
    torch.save(model.state_dict(), "model.pth")
    print("\nModel saved as model.pth")

    # Guardamos los parámetros del scaler
    os.makedirs("artifacts", exist_ok=True)

    scaler_df = pd.DataFrame({
        "feature": df.drop("failure", axis=1).columns,
        "mean": scaler.mean_,
        "scale": scaler.scale_
    })
    scaler_df.to_csv("artifacts/scaler_parameters.csv", index=False)

    print("Scaler parameters saved in artifacts/scaler_parameters.csv")


if __name__ == "__main__":
    main()