# heat_pinn_2D.py
# 
# A simple physics-informed neural network (PINN) using PyTorch
# to solve the 2D heat equation
#   u_t = alpha * (u_xx + u_yy),
# where x,y\in[0,1] and t\in[0,1], and the boundary conditions are
#   u(0, t) = u(1, 0) = 0
#   u(x, y, 0) = sin(pi*x)*sin(pi*y)
# The analytical solution is:
#   u(x,t) = \exp{-2*alpha*pi^2*t} * sin(pi*x) * sin(pi*y)
#
#
# Copyright (c) 2026 Rashid Vladimir Williams-Garcia, Atelier Markov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import torch
import os
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # <-- Required for 3D projections

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Step 1. Define the neural network
# ---------------------------------------------------------------------------

# We use a fully-connected network to approx the soln u(x,t)
class HeatPINN(nn.Module):
    def __init__(self, hidden_layers=4, nodes_per_layer=32):
        #to initialize, call init method from parent class nn.Module
        #inherit net, apply, _init_weights methods
        super(HeatPINN, self).__init__()
        layers = []

        # inputs representing (x,y,t) to the first fully-connected hidden layer
        layers.append(nn.Linear(3, nodes_per_layer))    #y=w1x+w2x+w3t+b, where the w's and b's are learnable
        #append nonlinear activation layer object:
        layers.append(nn.Tanh())    #tanh(y)

        #continue creating the remaining hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(nn.Tanh())

        #single output feature: u(x,y,t)
        layers.append(nn.Linear(nodes_per_layer, 1))
        self.net = nn.Sequential(*layers)    #wrap the whole net into one module, net
        #note: nn.Sequential(layers) passes the list itself as one argument,
        #while the *layers unpacks layers, e.g., layers = [l1,l2,l3]
        #nn.Sequential(*layers) = nn.Sequential(l1,l2,l3)

        #Xavier initialization:
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #Glorot uniform method: draws from a uniform distro in +/-\sqrt(6/(n_in+n_out))
            nn.init.xavier_uniform_(module.weight)
            #this is done to avoid vanishing/exploding gradients:
            # shrinking activations -> very slow training
            # growing activations -> unstable training
            #happens when variance of weights is not tuned to layer sizes...

            #initial zero biases:
            nn.init.zeros_(module.bias)

    #automatically called when using model as a callable
    def forward(self, x, y, t):
        inputs = torch.cat([x,y,t], dim=1)  #Nx3 tensor
        return self.net(inputs) #Nx1 tensor (the predicted u)

# ---------------------------------------------------------------------------
# Step 2. Define the physics-informed loss functions
# ---------------------------------------------------------------------------

class HeatPINNLoss:
    def __init__(self, alpha=1, device='cpu'):
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

    def pde_residual(self, model, x, y, t):
        #compute the residual R = u_t - alpha*(u_xx+u_yy)
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        u = model(x, y, t)  #automatically calls forward pass

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                  create_graph=True, retain_graph=True)[0]
        
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_x),
                                  create_graph=True, retain_graph=True)[0]

        residual = u_t - self.alpha * (u_xx + u_yy)
        return torch.mean(residual**2)

    def boundary_condition(self, model, x_bc, y_bc, t_bc):
        u_pred = model(x_bc, y_bc, t_bc)
        return torch.mean(u_pred**2)

    def initial_condition(self, model, x_ic, y_ic, t_ic):
        u_pred = model(x_ic, y_ic, t_ic)
        u_true = torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)
        return torch.mean((u_pred - u_true)**2)
    
# ---------------------------------------------------------------------------
# Step 3. Train the PINN
# ---------------------------------------------------------------------------

def train_pinn():
    #config compute devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #hyperparameters
    alpha = 1 #thermal diffusivity
    n_epochs = 10000
    lr = 0.001  #learning rate
    n_collocation = 2000    #number of points where the PDE is enforced
    n_boundary = 100 #number of points on the boundary
    n_initial = 100 #number of points at t=0

    #model initialization
    model = HeatPINN(hidden_layers=4, nodes_per_layer=32).to(device)
    loss_function = HeatPINNLoss(alpha=alpha, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #training data (collocation points)
    #randomly sampled inside domain to enforce the PDE
    x_pde = torch.rand(n_collocation, 1, device=device) #x in [0,1]
    y_pde = torch.rand(n_collocation, 1, device=device) #y in [0,1]
    t_pde = torch.rand(n_collocation, 1, device=device) #t in [0,1]

    #randomly sampled on the boundaries for all t
    #left edge, x=0
    x_bc_left = torch.zeros(n_boundary, 1, device=device)
    y_bc_left = torch.rand(n_boundary, 1, device=device)
    t_bc_left = torch.rand(n_boundary, 1, device=device)

    #right edge, x=1
    x_bc_right = torch.ones(n_boundary, 1, device=device)
    y_bc_right = torch.rand(n_boundary, 1, device=device)
    t_bc_right = torch.rand(n_boundary, 1, device=device)

    #bottom edge, y=0
    x_bc_bottom = torch.rand(n_boundary, 1, device=device)
    y_bc_bottom = torch.zeros(n_boundary, 1, device=device)
    t_bc_bottom = torch.rand(n_boundary, 1, device=device)

    #top edge, y=1
    x_bc_top = torch.rand(n_boundary, 1, device=device)
    y_bc_top = torch.ones(n_boundary, 1, device=device)
    t_bc_top = torch.rand(n_boundary, 1, device=device)

    #combined:
    x_bc = torch.cat([x_bc_left, x_bc_right, x_bc_bottom, x_bc_top], dim=0)
    y_bc = torch.cat([y_bc_left, y_bc_right, y_bc_bottom, y_bc_top], dim=0)
    t_bc = torch.cat([t_bc_left, t_bc_right, t_bc_bottom, t_bc_top], dim=0)

    #randomly sampled at t=0, for all x, y
    x_ic = torch.rand(n_initial, 1, device=device)
    y_ic = torch.rand(n_initial, 1, device=device)
    t_ic = torch.zeros(n_initial, 1, device=device)

    #Training Loop:
    loss_history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()   #reset gradients

        #forward passes to compute the loss
        loss_pde = loss_function.pde_residual(model, x_pde, y_pde, t_pde)
        loss_bc = loss_function.boundary_condition(model, x_bc, y_bc, t_bc)
        loss_ic = loss_function.initial_condition(model, x_ic, y_ic, t_ic)

        #combined loss (weights to be tuned)
        loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic

        #backpropagation: compute gradients and update weights
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch%1000==0:
            print(f"Epoch {epoch:5d} | Loss:{loss.item():.6f} (PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, IC: {loss_ic.item():.6f})")

    print("Training completed!")

    return model


# ---------------------------------------------------------------------------
# Visualize and validate
# ---------------------------------------------------------------------------

def visualize(model, device, alpha):
    model.eval()

    # --- Setup ---
    x_grid = torch.linspace(0, 1, 50, device=device)
    y_grid = torch.linspace(0, 1, 50, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    times = np.linspace(0, 0.3, 40)


    X_OFFSET = 1.2

    # --- Initial figure ---
    t_flat = torch.full_like(x_flat, times[0])

    with torch.no_grad():
        u_pred_init = model(x_flat, y_flat, t_flat).reshape(50, 50).cpu().numpy()

    u_true_init = np.exp(-2 * alpha * np.pi**2 * times[0]) * np.sin(np.pi * X_np) * np.sin(np.pi * Y_np)

    fig = go.Figure(
        data=[
            go.Surface(
                z=u_pred_init, x=X_np, y=Y_np,
                colorscale='Viridis', cmin=0, cmax=1,
                name='PINN', showscale=True
            )#,
            #go.Surface(
            #    z=(u_true_init-u_pred_init)**2, x=X_np + X_OFFSET, y=Y_np,
            #    colorscale='Plasma', cmin=0, cmax=1,
            #    name='Analytical', opacity=0.7, showscale=True
            #)
        ]
    )

    # --- Build frames and slider steps ---
    frames = []
    slider_steps = []
    for t_val in times:
        t_flat = torch.full_like(x_flat, t_val)

        #inference on trained model:
        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat).reshape(50, 50).cpu().numpy()

        u_true = np.exp(-2 * alpha * np.pi**2 * t_val) * np.sin(np.pi * X_np) * np.sin(np.pi * Y_np)

        frames.append(go.Frame(
            data=[
                go.Surface(z=u_pred)#,
                #go.Surface(z=(u_true-u_pred)**2)
            ],
            name=f't={t_val:.3f}',
            traces=[0, 1]
        ))

        slider_steps.append(dict(
            method="animate",
            # Label shown on the slider (e.g., "t = 0.05")
            label=f"t = {t_val:.2f}", 
            args=[
                [f"t={t_val:.3f}"],  # Must match the 'name' in your frames
                dict(
                    mode="immediate",
                    frame=dict(duration=100, redraw=True),
                    transition=dict(duration=0)
                )
            ]
        ))

    fig.frames = frames

    # --- Add play/pause buttons ---
    fig.update_layout(
        sliders=[dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},  # Padding from the plot
        steps=slider_steps
    )],
        title='PINN Solution – 2D Heat Equation',
        scene=dict(
            xaxis=dict(title='x', range=[0, 1], autorange=False),  # extend x-range
            yaxis=dict(title='y', range=[0, 1], autorange=False),
            zaxis=dict(title='u(x, y, t)', range=[0, 1], autorange=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),  # wider to fit both
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
         ),
        updatemenus=[dict(
            type='buttons',
            buttons=[
                dict(label='Play', method='animate',
                    args=[None, dict(frame=dict(duration=100, redraw=True),
                                    fromcurrent=True)]),
                dict(label='Pause', method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')])
            ]
        )]
    )

    fig.show()
    fig.write_html("pinn_2d_animation.html")

# ---------------------------------------------------------------------------
# Model saving and loading functions
# ---------------------------------------------------------------------------

def save_model(model, filepath="heat_pinn.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath="heat_pinn.pth"):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location='cpu', weights_only=True))
        print(f"Model loaded from {filepath}")
    else:
        print(f"File {filepath} not found")
    return model



if __name__ == "__main__":
    filepath="heat_pinn.pth"
    if os.path.exists(filepath):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = HeatPINN(hidden_layers=4, nodes_per_layer=32).to(device)
        model = load_model(model, filepath=filepath)
        visualize(model, device, 1)
    else:
        model = train_pinn()
        save_model(model)
