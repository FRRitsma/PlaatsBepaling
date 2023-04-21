# %%
import matplotlib.pyplot as plt
import numpy as np
from hilbert.hilbert import HilbertCoder # Required import for the pickle import of hilbert coder
import pickle
from utilities import loss_function
from sklearn.metrics import mean_absolute_error
from visualization.visualization import display_houses_facilities
from reinforcement.reinforcement import (
    filter_state_and_loss,
    generate_branch,
    proportional_sample,
)
from reinforcement.fitting import fit_model


# HARDCODES:
N_DIMENSIONS = 9
N_START_SAMPLES = 10
N_INITIALIZATION_SAMPLES = 200
EXPLORATION_RATE = .1

# Read data:
with open("hilbertcoder.pkl", "rb") as fs:
    hilbertcoder = pickle.load(fs)
house_locations = np.load("house_locations.npy")

# CHAPTER 1: Initialize model with random data:
# Step 1: collect all available states
ALL_AVAILABLE_STATES = np.sort(hilbertcoder.values, axis=0)
state_selection_permutation = np.random.permutation(ALL_AVAILABLE_STATES.shape[0])
state_subset = ALL_AVAILABLE_STATES[state_selection_permutation[:N_START_SAMPLES]]

# Step 2: Create an amount of random states:
random_states = np.random.choice(
    state_subset.ravel(), N_INITIALIZATION_SAMPLES * N_DIMENSIONS
)
random_states = random_states.reshape(N_INITIALIZATION_SAMPLES, N_DIMENSIONS)
random_states = -np.sort(-random_states, axis=1)

# Step 3: Get value/loss of each random state:
random_loss = loss_function(house_locations, hilbertcoder.decoding(random_states))

# Step 4: Fit simple model:
x, y = filter_state_and_loss(random_states, random_loss)
model = fit_model(x, y, 5)

#%% 
from reinforcement.better_reinforcement import score_branches, optimize_branches

a = optimize_branches(state_subset, N_DIMENSIONS, model, 100)





# %%

#  CHAPTER 2: Use model to perform reinforcement learning:
for i in range(N_START_SAMPLES, 1000, 50):
    # Proportionally sample better scoring states
    for seed in proportional_sample(state_subset, 50, N_DIMENSIONS, model):
        seed_state = -np.ones([1, N_DIMENSIONS])
        seed_state[0, 0] = seed
        x_new = generate_branch(x, state_subset, model, EXPLORATION_RATE, seed_state)
        y_new = np.hstack(
            [
                loss_function(house_locations, hilbertcoder.decoding(x_row))
                for x_row in x_new
            ]
        )
        x_new = np.vstack(x_new)
        x = np.vstack([x, x_new])
        y = np.hstack([y, y_new])

    # Print progress:
    plot_density = int(1e4)
    x_plot = -np.ones([plot_density, N_DIMENSIONS])
    x_plot[:, 0] = np.linspace(0, 1, plot_density)
    y_plot = model.predict(x_plot)

    y_scatter = y[x[:, 1] == -1]
    x_scatter = x[x[:, 1] == -1, 0]

    # Display state/value distribution of first state:
    plt.scatter(x_scatter, y_scatter, s=3)
    plt.plot(x_plot[:, 0], y_plot, ":")
    plt.xlabel("Placement of facility no. 1")
    plt.ylabel("Value of placement")
    plt.title(f"Current best value: {np.min(y)}")
    plt.show()

    # Display current distribution of facilities:
    x_best = x[y == np.min(y), :]
    x_best = x_best[-1, :]
    display_houses_facilities(hilbertcoder.decoding(x_best), house_locations)

    # Retrain model:
    x, y = filter_state_and_loss(x, y)
    model = fit_model(x, y, model.n_estimators)

    # Add new states:
    n_used_states = min(i, 250)
    state_subset = ALL_AVAILABLE_STATES[state_selection_permutation[:n_used_states]]

