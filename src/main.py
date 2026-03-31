
from utils import calculate_mean, calculate_variance, normalize_data


if __name__ == "__main__":
    data = [10, 20, 30, 40, 50]

    print("Original Data:", data)

    mean = calculate_mean(data)
    variance = calculate_variance(data)
    normalized = normalize_data(data)

    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Normalized Data: {normalized}")
