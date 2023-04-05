import numpy as np


def proportional_sample(
    states: np.ndarray, n: int, n_dimensions: int, model
) -> np.ndarray:
    # Determine size of choice:
    n = min(n, states.shape[0])

    # Determine value of current
    scoring_data = -np.ones([len(states), n_dimensions])
    scoring_data[:, 0] = states.flatten()
    scores = model.predict(scoring_data)

    # Sorting by score:
    idx = np.argsort(-scores)
    states = states[idx]

    # Scoring the states:
    probability = np.arange(len(states)) + 1
    probability = probability / np.sum(probability)

    return np.random.choice(states.flatten(), n, p=probability, replace=False)


def augment_state(states: np.ndarray) -> np.ndarray:
    length, dim = states.shape[0], states.shape[1]
    states = np.vstack([states] * (dim - 1))

    for i, idx in enumerate(range(length, length * (dim - 1), length)):
        i = i + 1
        states[idx : idx + length, i:-1] = -1

    return states


def filter_state_and_loss(
    states: np.ndarray, loss: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the best performing state in a list of states with duplicates. Removes duplicates.

    Args:
        states (np.ndarray): _description_
        loss (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """

    assert (
        states.shape[0] == loss.shape[0]
    ), "Amount of states must equal amount of losses"

    bundled_state_loss = np.hstack([states, loss.reshape(-1, 1)])
    bundled_state_loss = augment_state(bundled_state_loss)
    bundled_state_loss = bundled_state_loss[np.lexsort(bundled_state_loss.T), :]
    _, keep_index = np.unique(bundled_state_loss[:, :-1], axis=0, return_index=True)
    return bundled_state_loss[keep_index, :-1], bundled_state_loss[keep_index, -1]


def filter_available_states(
    available_states: np.ndarray, new_state: np.ndarray
) -> np.ndarray:
    ceiling = np.min(abs(new_state))
    return available_states[available_states <= ceiling]


def add_random_state(
    available_states: np.ndarray,
    new_state: np.ndarray,
) -> np.ndarray:
    # Early exit if state is finished:
    if not -1 in new_state:
        return new_state
    # Prevent external overwrite:
    new_state = new_state.copy()
    # Find un-initialized state:
    idx = np.where(new_state == -1)[1][0]
    # Randomly assign a state:
    new_state[0, idx] = np.random.choice(
        filter_available_states(available_states, new_state)
    )
    return new_state


def select_random_row(array: np.ndarray):
    return array[np.random.randint(array.shape[0]), :].reshape(1, -1)


def find_optimal_state(
    available_states: np.ndarray,
    new_state: np.ndarray,
    model,
) -> np.ndarray:
    # Early exit if state is finished:
    if not -1 in new_state:
        return new_state
    idx = np.where(new_state == -1)[1][0]

    # Filter available states:
    available_states = filter_available_states(available_states, new_state)

    # Find optimal state via the model:
    test_x = -np.ones([available_states.shape[0], new_state.shape[1]])
    test_x[:, :idx] = new_state[:, :idx]
    test_x[:, idx] = available_states
    test_y = model.predict(test_x)

    # Select best row:
    best_y = np.min(test_y)
    test_x = test_x[test_y <= best_y, :]
    best_x = select_random_row(test_x)

    return best_x


# Choose optimal state or explore:
def generate_branch(
    previous_states: np.ndarray,
    available_states: np.ndarray,
    model,
    EXPLORATION_RATE: float,
    new_state: np.ndarray,
) -> np.ndarray:
    if -1 in new_state:
        # Find the optimal state:
        optimal_state = find_optimal_state(
            available_states,
            new_state,
            model,
        )
        # Add random branch:
        if np.random.rand() < EXPLORATION_RATE:
            random_state = add_random_state(available_states, new_state)
            return generate_branch(
                previous_states,
                available_states,
                model,
                EXPLORATION_RATE,
                optimal_state,
            ) + generate_branch(
                previous_states,
                available_states,
                model,
                EXPLORATION_RATE,
                random_state,
            )

        # Return regular branch:
        else:
            return generate_branch(
                previous_states,
                available_states,
                model,
                EXPLORATION_RATE,
                optimal_state,
            )

    return [new_state]
