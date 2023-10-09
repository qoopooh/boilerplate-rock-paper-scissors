"""The example function below keeps track of the opponent's history and plays whatever
the opponent played two plays ago. It is not a very good player
so you will need to change the code to pass the challenge.
"""
import time

import numpy as np
import tensorflow as tf


HISTORY_LENGTH = 26
# Create a dictionary to map classes to labels
class_to_label = {"R": 0, "P": 1, "S": 2}
label_to_class = {v: k for k, v in class_to_label.items()}
start_time = time.time()
predicted_history = []


# Define the neural network model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            3, 8, input_length=(HISTORY_LENGTH - 2)
        ),  # Embedding layer to represent the history
        tf.keras.layers.Flatten(),  # Flatten the embedding output
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(
            3, activation="softmax"
        ),  # Output layer with 3 classes (rock, paper, scissors)
    ]
)


def player(prev_play, opponent_history: list[str] = []) -> str:
    """Playing"""
    global predicted_history
    if prev_play:
        opponent_history.append(prev_play)
        match_history = _create_match_history(
            opponent_history,
            predicted_history,
        )
    else:
        # New opponent
        global model
        model = tf.keras.models.clone_model(model)
        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        opponent_history, predicted_history, match_history = [], [], []

    next_move, prev = predict(match_history)
    if prev != prev_play:
        # Train model
        train_model(match_history)

    predicted_history.append(win_with(next_move))

    return win_with(next_move)


def train_model(match_history: list[str]) -> None:
    """Train model"""
    if len(match_history) < HISTORY_LENGTH:
        return

    # Train the model
    train_data = match_history[-HISTORY_LENGTH:-2]
    print(len(train_data), train_data)
    history = _history_to_array(train_data)
    batch = np.expand_dims(history, axis=0)  # Add a batch dimension
    out = class_to_label[match_history[-2]]
    y = np.array([out])  # The target label for the current prediction
    model.fit(batch, y, epochs=5)


def predict(match_history: list[str]):
    """Make a prediction"""
    if len(match_history) < HISTORY_LENGTH:
        prev = match_history[-2] if match_history else "R"
        return _dummy_next_move(prev, match_history), prev

    count = len(predicted_history)
    next_move = _model_predict(match_history[-(HISTORY_LENGTH - 2) :])
    # prev = _model_predict(match_history[-HISTORY_LENGTH:-2])
    if count % 4 == 0:
        prev = _model_predict(match_history[-HISTORY_LENGTH:-2])
    else:
        prev = match_history[-2]

    if prev != match_history[-2]:
        print(
            "Wrong prediction: {}".format(count),
            next_move,
            prev,
            match_history[-10:],
        )
    elif count % HISTORY_LENGTH == 0:
        print(elapsed_time(), count, next_move, prev, match_history[-10:])

    return next_move, prev


def _create_match_history(
    opponent_history: list[str], prediction_history: list[str]
) -> list[str]:
    match_history = []
    oppo = opponent_history[-(HISTORY_LENGTH // 2) :]
    pred = prediction_history[-(HISTORY_LENGTH // 2) :]
    for idx, item in enumerate(oppo):
        match_history.append(item)  # opponent
        if len(pred) > idx:
            match_history.append(pred[idx])  # our prediction

    # print("oppo", oppo)
    # print("pred", pred)
    # print("match_history", match_history)
    return match_history


def _model_predict(match_history: list[str]) -> str:
    """Predicted next_move move"""

    history = _history_to_array(match_history)
    batch = np.expand_dims(history, axis=0)  # Add a batch dimension
    # Predict the next_move move based on the updated history
    next_move_probabilities = model.predict(batch)
    predicted_move_label = np.argmax(next_move_probabilities)
    # Map the predicted label back to the corresponding class
    return label_to_class[predicted_move_label]


def _history_to_array(history: list[str]):
    """Create an array"""
    label_history = []
    for item in history:
        label_history.append(class_to_label[item])
    return np.array(label_history)


def _dummy_next_move(play: str, match_history: list[str]) -> str:
    """dummy move before training"""
    if len(match_history) > 4 and match_history[-4] == play:
        # opponent will keep the same play
        return play

    if play == "R":
        return "S"

    if play == "P":
        return "R"

    return "P"


def elapsed_time() -> int:
    """Get running time"""
    return int(time.time() - start_time)


def win_with(play) -> str:
    """Return strategy to win"""
    if play == "S":
        return "R"

    if play == "P":
        return "S"

    return "P"
