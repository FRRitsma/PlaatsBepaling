# %%
from reinforcement import filter_available_states
import numpy as np

# Faster, non recursive method of testing/branching states
# Determine exploration rate


turns_left = 3
n_end_branches = 40
n_current_branches = 20


def compute_exploration_rate(
    turns_left: int,
    n_end_branches: int,
    n_current_branches: int,
) -> float:
    if n_end_branches < n_current_branches:
        return 0
    return (n_end_branches / n_current_branches) ** (1 / turns_left) - 1


# Stack a state and their available states:
def single_explorable_branch(
    branch: np.ndarray,
    available_states: np.ndarray,
) -> np.ndarray:
    assert branch.ndim == 2, "'new_state' must be two dimensional"

    # Early exit if state is finished:
    if not -1 in branch:
        return branch

    idx = np.where(branch == -1)[1][0]

    # Filter available states:
    available_states = filter_available_states(available_states, branch)

    # Find optimal state via the model:
    test_x = -np.ones([available_states.shape[0], branch.shape[1]])
    test_x[:, :idx] = branch[:, :idx]
    test_x[:, idx] = available_states

    return test_x


def multi_explorable_branch(
    branches: np.ndarray,
    available_states: np.ndarray,
    n_end_branches: int,
) -> np.ndarray:
    
    # Early exit:
    if not -1 in branches:
        return branches
    
    # TODO: find exploration rate
    turns_left = np.count_nonzero(branches[0,:] == -1)
    #n_current_branches = 

    # Sample random branches:
    random_branches = np.empty([0, branches.shape[1]])
    exploration_rate = 0.8
    while exploration_rate > 0:
        mask = np.random.rand(len(branches)) < exploration_rate
        random_branches = np.vstack((random_branches, branches[mask, :]))
        exploration_rate -= 1

    # Sample branches for optimization:
    optimizable_branches = np.vstack(
        tuple(
            single_explorable_branch(b.reshape(1, -1), available_states)
            for b in branches
        )
    )

    return optimizable_branches, random_branches


available_states = np.linspace(0, 1, 5)
new_state = -np.ones([2, 4])
new_state[:, 0] = np.linspace(0.25, 0.8, 2)


# n_current_branches * (exploration_rate ** turns_left) = n_predicted_branches
# (exploration_rate ** turns_left) = n_predicted_branches/n_current_branches
# exploration_rate ** turn_left = (n_predicted_branches/n_current_branches)
# exploration_rate = (n_predicted_branches/n_current_branches)**(1/turns_left)
