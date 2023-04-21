# %%

import numpy as np

# Faster, non recursive method of testing/branching states
# Determine exploration rate


turns_left = 3
n_end_branches = 40
n_current_branches = 20


def filter_available_states(
    available_states: np.ndarray, new_state: np.ndarray
) -> np.ndarray:
    ceiling = np.min(abs(new_state))
    return available_states[available_states <= ceiling]


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
    turns_left = np.count_nonzero(branches[0, :] == -1)
    # n_current_branches =

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


# %%


def expand_single_branch(
    available_states: np.ndarray,
    branch: np.ndarray,
) -> np.ndarray:
    assert branch.ndim == 1, "'branch' must be 1-dimensional"

    if not -1 in branch:
        return branch

    idx = np.where(branch == -1)[0][0]
    subset_available_states = filter_available_states(available_states, branch)
    expanded_branch = np.vstack((branch,) * len(subset_available_states))
    expanded_branch[:, idx] = subset_available_states

    return expanded_branch


def expand_multi_branches(
    available_states: np.ndarray,
    branches: np.ndarray,
) -> np.ndarray:
    assert branches.ndim == 2, "'branches' must be 2-dimensional"

    # Early exit if branches have been filled with data:
    if not -1 in branches:
        return branches

    # Create all combinations of current branches and possible branches:
    expanded_branches = np.vstack(
        tuple(expand_single_branch(available_states, branch) for branch in branches)
    )

    # Remove duplicate rows:
    expanded_branches = np.unique(expanded_branches, axis=0)

    return expanded_branches


def score_branches(
    branches: np.ndarray,
    model,
) -> tuple[np.ndarray, np.ndarray]:
    assert branches.ndim == 2, "'branches' must be 2-dimensional"
    assert (
        getattr(model, "predict", None) is not None
    ), "'model' must have method predict method"

    score = model.predict(branches)

    return score


def select_best_branches(branches: np.ndarray, scores: np.ndarray, n_best: int) -> np.ndarray:
    """Return the rows in branches with the best scores. Best scores are lowest scores.

    Args:
        branches (np.ndarray): Branches
        scores (np.ndarray): Scores
        n_best (int): Choose this amount of best samples

    Returns:
        np.ndarray: Best branches returned as single array
    """

    assert n_best > 0, "'n_best' must be a positive integer"
    assert branches.ndim == 2, "'branches' must be 2-dimensional"
    idx = np.argsort(scores)

    return branches[idx[:n_best], :]


def select_random_rows(array: np.ndarray, choose_n: int) -> np.ndarray:
    """Randomly select rows from 2-dimensional array

    Args:
        array (np.ndarray): From which rows are selected
        choose_n (int): Amount of rows to be selected

    Returns:
        np.ndarray: Resulting rows, returned as single array
    """
    assert choose_n > 0, "'choose_n' must be a positive integer"
    assert array.ndim == 2, "'array' must be 2-dimensional"

    return array[np.random.permutation(array.shape[0])[:choose_n], :]


# Select branches:
# 1 - Select best branches overall:

# 2 - Select random states:

# 3 - Select best branches randomly initialized:

# 4 - Add new random branches:


# Explore, keep #n best branches:
def optimize_branches(
    available_states: np.ndarray,
    n_states: int,
    model,
    n_optimal_states: int,
    n_random_states: int = 5,  # TODO: Fix awful hardcode
) -> np.ndarray:
    # Enforce correct input:
    assert n_states > 0, "'n_states' must be a positive integer"
    assert n_optimal_states > 0, "'n_optimal_states' must be a positive integer"
    assert n_random_states > 0, "'n_random_states' must be a positive integer"
    assert (
        getattr(model, "predict", None) is not None
    ), "'model' must have method predict"

    # Initialization
    available_states = available_states.flatten()

    # Initialization of optimal branches:
    optimal_branches = -np.ones((len(available_states), n_states))
    optimal_branches[:, 0] = available_states

    # Initialization of random branches:
    random_branches = select_random_rows(optimal_branches, n_random_states)

    assert optimal_branches.shape[1] == n_states, "FIRST POINT OF FAILURE"

    # Select best branches continuously:
    while -1 in optimal_branches:
        
        print(optimal_branches.shape)

        # Score all current branches
        all_scores = score_branches(
            np.vstack((optimal_branches, random_branches)),
            model,
        )

        optimal_score = all_scores[: -random_branches.shape[0]]
        
        # random_score = all_scores[-random_branches.shape[0]]

        # Select from optimal and random branches:
        optimal_branches = select_best_branches(
            optimal_branches,
            optimal_score,
            n_optimal_states,
        )
        print(optimal_branches.shape)

        # assert optimal_branches.shape[1] == n_states, "SECOND POINT OF FAILURE"

        # random_branches = optimize_random_branches(
        #     random_branches,
        #     random_score,
        # )
        # assert optimal_branches.shape[1] == n_states, "THIRD POINT OF FAILURE"

        # Create candidate optimal branches:
        optimal_branches = expand_multi_branches(
            available_states,
            optimal_branches,
        )

        # # Create candidate random branches:
        # candidate_random_branches = expand_multi_branches(
        #     available_states,
        #     random_branches,
        # )
        # random_branches = np.hstack(
        #     (
        #         select_random_rows(
        #             np.vstack((optimal_branches, candidate_random_branches)),
        #             n_random_states,
        #         ),
        #         random_branches,
        #     )
        # )

    

    return optimal_branches


# if __name__ == "__main___":
#     available_states = np.linspace(0, 1, 5)
#     new_state = -np.ones([2, 4])
#     new_state[:, 0] = np.linspace(0.25, 0.8, 2)
#     branches = new_state.copy()
#     model = None

#     print(optimize_branches(branches, model, 10))
# %%
random_branches = -np.ones((3, 5))
available_states = np.random.rand(5)
random_branches[:, 0] = np.random.rand(3) + 2
random_branches[:, 1] = np.random.rand(3) + 1


def optimize_random_branches(
    random_branches: np.ndarray,
    random_score: np.ndarray,
) -> np.ndarray:
    # Enforce correct input:
    assert random_branches.ndim == 2, "'random_branches' must be 2-dimensional"
    assert random_score.ndim == 1, "'random_score' must be 1-dimensional"
    assert (
        random_branches.shape[0] == random_score.shape[0]
    ), "branches and score must have the same length"

    # Early exit if branches have been filled with data:
    if not -1 in random_branches:
        return random_branches

    # Select which axis to keep:
    idx = np.where(random_branches[0, :] == -1)[0][0] - 1
    assert idx > -1, "'idx' can not be zero"

    # Logic: keeping the optimal version of each random branch:
    random_concatenated = np.hstack([random_branches, random_score.reshape(-1, 1)])
    # Bundle branches and loss:
    random_concatenated = random_concatenated[np.lexsort(random_concatenated.T), :]
    # Sort lexicographically:
    random_concatenated = random_concatenated[np.argsort(random_score), :]
    # Keep first unique row:
    _, keep_index = np.unique(random_concatenated[:, :idx], axis=0, return_index=True)

    return random_concatenated[keep_index, :-1]


# Select best of random branches
