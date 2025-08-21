#from lib.base_nn import squawk
import lib.base_nn as BNN
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

ian_model = BNN.Ia_NN()
model = BNN.Toy_NN()
model.plot()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
losses = [None]*epochs
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(model.input)
    loss = criterion(outputs, model.output)
    loss.backward()
    optimizer.step()

    losses[epoch] = loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


fig, ax = plt.subplots()
ax.plot(np.linspace(0,epochs,epochs), losses)
fig.savefig("toy_losses.png")

res = model.forward(torch.tensor(np.array([[8,0,1],[5,0,1]], dtype=np.float32)))
print(res)
#        self.input = torch.tensor(np.column_stack((sleep,race,prep)))

