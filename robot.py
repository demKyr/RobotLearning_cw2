# Import some installed modules.
import numpy as np

# TO BE REMOVED
import graphics


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
        # CUSTOM DATA
        self.planned_actions = None
        self.planning_horizon = 100
        self.needs_planning = True
        self.plan_timestep = 0

    # Function to compute the next action, during the training phase.
    def next_action_training(self, state):
        # For now, just a random action. Try to do better than this!
        next_action = self.random_action()

        # angle = np.deg2rad(180)
        # action_x = self.max_action * np.cos(angle)
        # action_y = self.max_action * np.sin(angle)
        # next_action = np.array([action_x, action_y])
        # next_action = self.goal_state - self.state

        reset = False
        return next_action, reset

    # Function to compute the next action, during the testing phase.
    def next_action_testing(self, state):
        # For now, just a random action. Try to do better than this!
        # next_action = self.random_action()

        if(self.needs_planning):
            # self.planned_actions = self.planning_random_shooting(state,self.planning_horizon)
            self.planned_actions = self.planning_cross_entropy(state,self.planning_horizon, K = 0.03, N = 100)
            self.needs_planning = False
        
        if(self.plan_timestep < self.planning_horizon):
            next_action = self.planned_actions[self.plan_timestep]
            self.plan_timestep += 1
        else:
            self.needs_planning = True
            next_action = np.array([0.0, 0.0])
            self.plan_timestep = 0

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



    # CUSTOM FUNCTIONS
    def planning_random_shooting(self, state, planning_horizon):
        best_path_dist = 9999
        best_planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
        best_planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
        for k in range(100):
            # Create an empty array to store the planned actions.
            planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
            # Create an empty array to store the planned states.
            planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
            # Set the initial state in the planning to the robot's current state.
            pred_planning_state = state
            # Loop over the planning horizon.
            for i in range(planning_horizon):
                # Choose a random action.
                angle = np.random.uniform(0, 2 * 3.141592)
                action = self.convert_angle_to_action(angle)
                # Simulate the next state using the model.
                pred_planning_state, uncertainty = self.model.predict(pred_planning_state, action)
                # planning_state = self.model(planning_state, action)
                # Add this action to the array of planned actions.
                planned_actions[i] = action
                # Add this state to the array of planned states.
                planned_states[i] = pred_planning_state
            # Create a path for these states, add it to the list of paths to be drawn.
            path = graphics.Path(planned_states, (255, 200, 0), 2, 0)
            self.paths_to_draw.append(path)
            # COMPARE TO BEST PATH
            dist = np.sqrt((planned_states[-1][0] - self.goal_state[0])**2+(planned_states[-1][1] - self.goal_state[1])**2)
            if(dist < best_path_dist):
                best_path_dist = dist
                best_planned_actions = planned_actions
                best_planned_states = planned_states
        # Return the array of best actions
        path = graphics.Path(best_planned_states, (255, 0, 0), 2, 0)
        self.paths_to_draw.append(path)
        return best_planned_actions


    def planning_cross_entropy(self, state, planning_horizon, K = 0.10, N = 100):
        best_planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
        best_planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
        color = 250
        mean = np.zeros(planning_horizon, dtype=np.float32)
        cov = np.zeros([planning_horizon, planning_horizon], dtype=np.float32)
        np.fill_diagonal(cov, 3.141592**2)
        for j in range(5):
            best_path_dist = 9999
            best_planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
            best_planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
            angles_and_dist = [{'dist':0, 'angles':np.zeros(planning_horizon, dtype=np.float32)} for l in range(N)]
            for k in range(N):
                # Create an empty array to store the planned actions.
                planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
                # Create an empty array to store the planned states.
                planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
                # Set the initial state in the planning to the robot's current state.
                pred_planning_state = state
                # Loop over the planning horizon.
                angles = np.random.multivariate_normal(mean, cov)
                for i in range(planning_horizon):
                    angle = angles[i]
                    # Choose an action.
                    # angle = np.random.uniform(-3.141592, 3.141592)
                    action = self.convert_angle_to_action(angle)
                    # Simulate the next state using the model.
                    pred_planning_state, uncertainty = self.model.predict(pred_planning_state, action)
                    # Add this action to the array of planned actions.
                    planned_actions[i] = action
                    # Add this state to the array of planned states.
                    planned_states[i] = pred_planning_state
                    # SAVE ACTION
                    angles_and_dist[k]['angles'][i] = angle
                # Create a path for these states, add it to the list of paths to be drawn.
                # path = graphics.Path(planned_states, (255, 200, 0), 2, 0)
                # self.paths_to_draw.append(path)
                # COMPARE TO BEST PATH
                dist = np.sqrt((planned_states[-1][0] - self.goal_state[0])**2+(planned_states[-1][1] - self.goal_state[1])**2)
                angles_and_dist[k]['dist'] = dist
                if(dist < best_path_dist):
                    best_path_dist = dist
                    best_planned_actions = planned_actions
                    best_planned_states = planned_states
            # CHANGE DISTRIBUTION'S PARAMS
            sorted_angles_and_dist = sorted(angles_and_dist, key=lambda d: d['dist']) 
            selected_angles = []
            for idx in range (int(K*N)):
                selected_angles.append(sorted_angles_and_dist[idx]['angles'])
            selected_angles = np.array(selected_angles)
            mean = np.mean(selected_angles, axis=0)
            np.fill_diagonal(cov, np.var(selected_angles, axis=0))
            # Return the array of best actions
            path = graphics.Path(best_planned_states, (255, color, 0), 2, 0)
            color -= 50
            self.paths_to_draw.append(path)
        self.finished = True
        return best_planned_actions

    # Function to convert a scalar angle, to an action parameterised by a 2-dimensional [x, y] direction.
    def convert_angle_to_action(self, angle):
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        return action
