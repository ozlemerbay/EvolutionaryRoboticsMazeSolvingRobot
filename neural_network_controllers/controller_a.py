import numpy as np

class ControllerA:
    """feedforward nn with no hidden layers"""
    def __init__(self, num_inputs=5, num_outputs=2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weights = np.zeros((self.num_inputs, self.num_outputs))
        self.biases = np.zeros(self.num_outputs)
        self.genome = (self.num_inputs * self.num_outputs) + self.num_outputs

    def set_weights(self, genome):
        bias_start_idx = self.num_inputs * self.num_outputs
        self.weights = np.array(genome[:bias_start_idx]).reshape((self.num_inputs, self.num_outputs))
        self.biases = np.array(genome[bias_start_idx:])

    def forward(self, inputs):
        """
        inputs: [forward, left, right, distance-to-goal, angle-to-goal]
        output: [left_speed, right_speed]
        """
        inputs = np.array(inputs)
        raw_outputs = np.dot(inputs, self.weights) + self.biases

        # tanh (-1 ,1)
        left_speed = np.tanh(raw_outputs[0])
        right_speed = np.tanh(raw_outputs[1])

        return left_speed, right_speed
