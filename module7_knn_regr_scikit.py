import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def main():
    try:
        # 1. Ask the user for input N (positive integer)
        N = int(input("Enter N (number of points, positive integer): "))
        if N <= 0:
            print("Error: N must be a positive integer.")
            return

        # 2. Ask the user for input k (positive integer)
        k = int(input("Enter k (number of neighbors, positive integer): "))
        if k <= 0:
            print("Error: k must be a positive integer.")
            return

        # Initialize Numpy arrays for data processing
        # x_data is 2D because scikit-learn features require a 2D array
        x_data = np.zeros((N, 1)) 
        y_data = np.zeros(N)

        # 3. Read N (x, y) points
        print(f"\nPlease enter the {N} points one by one:")
        for i in range(N):
            print(f"Point {i+1}:")
            x_val = float(input("  Enter x value: "))
            y_val = float(input("  Enter y value: "))
            # Insert data into numpy arrays
            x_data[i, 0] = x_val
            y_data[i] = y_val

        # 4. Ask the user for input X for prediction
        X_input = float(input("\nEnter X value to predict Y: "))
        X_pred = np.array([[X_input]])

        # 5. Check condition and execute ML computation
        if k > N:
            print(f"\nError: k ({k}) cannot be greater than N ({N}).")
        else:
            # Calculate and output the variance of labels
            label_variance = np.var(y_data)
            print(f"\nVariance of labels in the training dataset: {label_variance:.4f}")

            # Perform k-NN Regression using Scikit-learn
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_data, y_data)
            
            # Output the prediction result
            Y_pred = knn.predict(X_pred)
            print(f"The predicted result (Y) for X={X_input} using k-NN Regression is: {Y_pred[0]:.4f}")

    except ValueError:
        print("\nError: Invalid input type. Please enter numerical values as requested.")

if __name__ == "__main__":
    main()
