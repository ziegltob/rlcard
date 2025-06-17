import numpy as np

class ONNXAgent(object):
    def __init__(self, session, num_actions, state_shape):
        """
        Initialize ONNX agent
        
        Args:
            session (onnxruntime.InferenceSession): ONNX Runtime session
            num_actions (int): Number of possible actions
            state_shape (list): Shape of state input
        """
        self.session = session
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.use_raw = False

    def step(self, state):
        """
        Take a step
        
        Args:
            state (dict): Current state
            
        Returns:
            action (int): Chosen action
        """
        obs = state['obs'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_keys = np.array(list(legal_actions.keys()))
        
        # Create individual inputs for each legal action
        values = []
        for action_key in action_keys:
            # Create one-hot action vector
            action_features = np.zeros((1, self.num_actions), dtype=np.float32)
            action_features[0, action_key] = 1
            
            # Run inference for each action
            ort_inputs = {
                "obs": obs.reshape(1, -1),
                "actions": action_features
            }
            value = self.session.run(None, ort_inputs)[0]
            values.append(value[0])  # Store scalar value
        
        # Convert to numpy array for easier processing
        values = np.array(values)
        
        # Choose action with highest predicted value
        action_idx = np.argmax(values)
        action = action_keys[action_idx]
        
        return action

    def eval_step(self, state):
        """ 
        Same as step
        """
        return self.step(state), None