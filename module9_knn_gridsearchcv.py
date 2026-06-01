import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def main():
    # --- 1. Processing Training Data (TrainS) ---
    N = int(input("Enter N (positive integer for training pairs): "))
    
    # Initialize training data arrays using Numpy
    # X needs to be a 2D array (N, 1) for scikit-learn compatibility
    TrainX = np.empty((N, 1), dtype=float) 
    TrainY = np.empty(N, dtype=int)
    
    print(f"Please provide {N} (x, y) pairs for the training set:")
    for i in range(N):
        x = float(input(f"Train pair {i+1} - x value (real number): "))
        y = int(input(f"Train pair {i+1} - y value (non-negative integer): "))
        TrainX[i, 0] = x  # Data insertion
        TrainY[i] = y

    # --- 2. Processing Test Data (TestS) ---
    M = int(input("Enter M (positive integer for test pairs): "))
    
    # Initialize test data arrays using Numpy
    TestX = np.empty((M, 1), dtype=float)
    TestY = np.empty(M, dtype=int)
    
    print(f"Please provide {M} (x, y) pairs for the test set:")
    for i in range(M):
        x = float(input(f"Test pair {i+1} - x value (real number): "))
        y = int(input(f"Test pair {i+1} - y value (non-negative integer): "))
        TestX[i, 0] = x
        TestY[i] = y

    # --- 3. ML Computation & Hyperparameter Search ---
    # The requirement asks for 1 <= k <= 10. 
    # We use min(10, N) to ensure k does not exceed the number of training samples.
    max_k = min(10, N)
    param_grid = {'n_neighbors': range(1, max_k + 1)}
    
    knn = KNeighborsClassifier()
    
    # GridSearchCV automatically uses cross-validation.
    # We dynamically set 'cv' to avoid crashing if the user inputs a very small N (e.g., N < 5).
    cv_splits = min(5, N) 
    
    if cv_splits >= 2:
        # Perform hyperparameter search
        grid_search = GridSearchCV(knn, param_grid, cv=cv_splits, scoring='accuracy')
        grid_search.fit(TrainX, TrainY)
        
        # Extract the best model and best k
        best_k = grid_search.best_params_['n_neighbors']
        best_model = grid_search.best_estimator_
    else:
        # Fallback in case the user inputs N=1 (Cross-Validation requires at least 2 samples)
        best_k = 1
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(TrainX, TrainY)
        
    # --- 4. Evaluate and Output Results ---
    # Test on x values from TestS and compare with y values
    test_accuracy = best_model.score(TestX, TestY)
    
    print("\n--- Results ---")
    print(f"Best k: {best_k}")
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
