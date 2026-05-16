import numpy as np

def main():
    try:
        # Ask for input N
        N = int(input("Enter N (positive integer): "))
        if N <= 0:
            print("Error: N must be a positive integer.")
            return

        # Ask for input k
        k = int(input("Enter k (positive integer): "))
        if k <= 0:
            print("Error: k must be a positive integer.")
            return

        # Initialize numpy arrays to store the data
        x_data = np.zeros(N)
        y_data = np.zeros(N)

        # Ask for N (x, y) points
        print(f"Please provide {N} (x, y) points:")
        for i in range(N):
            x_val = float(input(f" Point {i+1} - Enter x value: "))
            y_val = float(input(f" Point {i+1} - Enter y value: "))
            x_data[i] = x_val
            y_data[i] = y_val

        # Ask for the target X input
        X_target = float(input("\nEnter the target X value to predict: "))

        # Verify if k <= N
        if k > N:
            print("Error: k cannot be greater than the number of points (N).")
            return

        # --- k-NN Regression calculation using Numpy ---
        
        # 1. Calculate distances from target X to all X points in the dataset
        # Since it's 1D, the distance is just the absolute difference
        distances = np.abs(x_data - X_target)

        # 2. Find the indices of the k nearest neighbors
        # np.argsort returns the indices that would sort the array
        nearest_indices = np.argsort(distances)[:k]

        # 3. Retrieve the corresponding Y values for those nearest neighbors
        nearest_y_values = y_data[nearest_indices]

        # 4. Calculate the predicted Y (mean of the nearest Y values)
        predicted_Y = np.mean(nearest_y_values)

        # Output the result
        print(f"\nThe predicted result (Y) of {k}-NN Regression is: {predicted_Y}")

    except ValueError:
        print("Error: Invalid input type. Please ensure you enter integers for N and k, and real numbers for x and y.")

if __name__ == "__main__":
    main()
