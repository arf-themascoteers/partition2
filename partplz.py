import torch
import torch.optim as optim

def print_partition(s, numbers):
    #print("Raw s",s)
    s_rounded = torch.tanh(s)
    #print("tanh(s)", s_rounded)

    subset_A = numbers[s_rounded > 0].tolist()
    subset_B = numbers[s_rounded <= 0].tolist()

    sum_A = sum(subset_A)
    sum_B = sum(subset_B)

    print("\nPartition:")
    print(f"Subset A: {subset_A}, Sum A: {sum_A}")
    print(f"Subset B: {subset_B}, Sum B: {sum_B}")

numbers = [24, 57, 16, 19, 81, 36, 69, 21, 24, 100, 42, 34, 9, 75, 45, 58, 24, 62, 45, 22, 74, 8, 88, 67, 56, 13, 27, 81, 31, 63, 59, 33, 50, 26, 44, 55, 70, 72, 68, 61, 50, 61, 17, 80, 38, 28, 30, 23, 12, 39, 95, 66, 71, 79, 85, 11, 74, 34, 57, 51, 82, 91, 90, 66, 16, 27, 68, 70, 83, 73, 1, 33, 14, 93, 29, 65, 2, 60, 37, 38, 41, 65, 25, 64, 94, 15, 79, 46, 10, 47, 76, 4, 48, 49, 77, 78, 86, 3, 98, 53, 20, 18, 32, 84, 96, 5, 7]
numbers = torch.tensor(numbers, dtype=torch.float32)
n = len(numbers)

s = torch.randn(n, dtype=torch.float32, requires_grad=True)

optimizer = optim.SGD([s], lr=0.001)
iterations = 700

print_partition(s, numbers)

for epoch in range(iterations):
    optimizer.zero_grad()
    #print(s)
    bin = torch.tanh(s)
    #print(bin)
    subset_sum = torch.sum(numbers * bin)
    norm = torch.norm(bin, p=1)
    energy = torch.abs(subset_sum) - norm
    print(torch.abs(subset_sum).item(), (norm).item())
    #print(subset_sum.item())
    energy.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{iterations} - Energy: {energy.item()}")

print_partition(s, numbers)