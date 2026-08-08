import numpy as np
from config import Config

class ControllerB:
    """feedforward nn with multiple hidden layers"""
    def __init__(self, hidden_layers, hidden_nodes, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        self.w1 = np.zeros((num_inputs, hidden_nodes))
        self.b1 = np.zeros(hidden_nodes)

        self.hidden_weights = []
        self.hidden_biases = []
        for _ in range(1, hidden_layers):
            self.hidden_weights.append(np.zeros((hidden_nodes, hidden_nodes)))
            self.hidden_biases.append(np.zeros(hidden_nodes))

        self.w_output = np.zeros((hidden_nodes, num_outputs))
        self.b_output = np.zeros(num_outputs)

        self.genome_len = (
            num_inputs * hidden_nodes
            + hidden_nodes
            + (hidden_layers - 1) * (hidden_nodes * hidden_nodes + hidden_nodes)
            + hidden_nodes * num_outputs
            + num_outputs
        )

    def set_weights(self, genome):
        idx = 0
        w1_end_idx = idx + self.num_inputs * self.hidden_nodes
        self.w1 = np.array(genome[idx:w1_end_idx]).reshape((self.num_inputs, self.hidden_nodes))
        idx = w1_end_idx

        b1_end_idx = idx + self.hidden_nodes
        self.b1 = np.array(genome[idx:b1_end_idx])
        idx = b1_end_idx

        for layer in range(1, self.hidden_layers):
            w_end_idx = idx + self.hidden_nodes * self.hidden_nodes
            self.hidden_weights[layer - 1] = np.array(genome[idx:w_end_idx]).reshape((self.hidden_nodes, self.hidden_nodes))
            idx = w_end_idx

            b_end_idx = idx + self.hidden_nodes
            self.hidden_biases[layer - 1] = np.array(genome[idx:b_end_idx])
            idx = b_end_idx

        w_out_end_idx = idx + self.hidden_nodes * self.num_outputs
        self.w_output = np.array(genome[idx:w_out_end_idx]).reshape((self.hidden_nodes, self.num_outputs))
        idx = w_out_end_idx

        self.b_output = np.array(genome[idx:])

    def forward(self, inputs):
        """
        inputs: [forward, left, right, distance-to-goal, angle-to-goal]
        output: [left_speed, right_speed]
        """
        inputs = np.array(inputs)

        hidden = np.dot(inputs, self.w1) + self.b1
        hidden = np.tanh(hidden)

        for w, b in zip(self.hidden_weights, self.hidden_biases):
            hidden = np.dot(hidden, w) + b
            hidden = np.tanh(hidden)

        raw_outputs = np.dot(hidden, self.w_output) + self.b_output
        left_speed = np.tanh(raw_outputs[0])
        right_speed = np.tanh(raw_outputs[1])

        return left_speed, right_speed
