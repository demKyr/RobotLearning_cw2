# Import some installed modules.
import numpy as np


# Define the Robot class
class Robot:

    # Initialisation function to create a new robot
    def __init__(self, model, max_action, goal_state):
        # STATE AND ACTION DATA
        # The maximum magnitude of the robot's action. Do not edit this.
        self.max_action = max_action
        # The goal state
        self.goal_state = goal_state
        # MODEL DATA
        self.model = model
        # VISUALISATION DATA
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []

    # Function to compute the next action, during the training phase.
    def next_action_training(self, state):
        # For now, just a random action. Try to do better than this!
        next_action = self.random_action()
        reset = False
        return next_action, reset

    # Function to compute the next action, during the testing phase.
    def next_action_testing(self, state):
        # For now, just a random action. Try to do better than this!
        next_action = self.random_action()
        return next_action

    # Function to compute the next action.
    def random_action(self):
        # Choose a random action.
        action_x = np.random.uniform(-self.max_action, self.max_action)
        action_y = np.random.uniform(-self.max_action, self.max_action)
        action = np.array([action_x, action_y])
        # Return this random action.
        return action

    def process_transition(self, state, action):
        self.model.update_uncertainty(state, action)
