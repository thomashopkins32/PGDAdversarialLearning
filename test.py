import torch

x = torch.tensor([1.], requires_grad=True)
y = x ** 2 + 1
z = 2 * y

z.backward()


print(y.grad)
print(x.grad)
