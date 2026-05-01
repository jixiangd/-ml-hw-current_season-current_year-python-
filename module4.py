def main():
    # 1. Ask for N (positive integer)
    n = int(input("Enter N (positive integer): "))
    
    # 2. Ask for N numbers one by one
    numbers = []
    for i in range(n):
        num = int(input(f"Enter number {i + 1}: "))
        numbers.append(num)
        
    # 3. Ask for X (integer)
    x = int(input("Enter X: "))
    
    # 4. Output the index (1 to N) or -1
    if x in numbers:
        # .index() returns 0-based index, so we add 1 to match the 1 to N requirement
        print(numbers.index(x) + 1)
    else:
        print("-1")

if __name__ == "__main__":
    main()
