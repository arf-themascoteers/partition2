import torch
import torch.optim as optim

def print_partition(s, numbers):
    s_rounded = torch.tanh(s)

    subset_A = numbers[s_rounded > 0].tolist()
    subset_B = numbers[s_rounded <= 0].tolist()

    sum_A = sum(subset_A)
    sum_B = sum(subset_B)

    print("\nPartition:")

    print(f"Sum A: {sum_A} {len(subset_A)}")
    print(f"Sum B: {sum_B} {len(subset_B)}")


numbers = torch.rand(100, dtype=torch.float32)
n = len(numbers)

s = torch.cat([torch.randn(80) - 2, torch.randn(20) + 2]).requires_grad_()


optimizer = optim.SGD([s], lr=0.01)
iterations = 70000

print_partition(s, numbers)

for epoch in range(iterations):
    optimizer.zero_grad()
    bin = torch.tanh(s)
    subset_sum = torch.sum(numbers * bin)
    norm = torch.norm(bin, p=1)
    energy = torch.abs(subset_sum) #- norm/100
    energy.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{iterations} - Energy: {energy.item()}")
        print(torch.abs(subset_sum).item(), (norm).item())

print_partition(s, numbers)