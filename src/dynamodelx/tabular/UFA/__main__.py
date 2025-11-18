from .ufa import UFA
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_wine, load_linnerud, load_iris
from sklearn.datasets import make_regression
from sklearn.datasets import load_breast_cancer
import numpy as np

def main(instance: UFA, X: np.ndarray, y: np.ndarray):
    ufa = instance(
        task='classification', 
        model_size='small', 
        input_dim=X.shape[1],
        output_dim= len(np.unique(y)),
        loss='cross_entropy_loss',
        device='cuda',
        custom_architecture=None, 
        weights_init='he', 
        hidden_activation='relu',
        optimizer='adam',
        return_metrics = True,
        auto_build=True,
        multiclass=True,
        uncertainty=False
        )
    
    performance = ufa.train(X, y, epochs=50, batch_size=32, learning_rate=0.01, val_size=0.3, test_size=0.2)
    out = ufa.predict(X[:1])
    print(f"Prediction output: {out}")
    ufa.save_model("trained_ufa_model.pth")
    

if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)

    main(UFA, X, y)