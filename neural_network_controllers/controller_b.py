import numpy as np

class ControllerB:
    """feedforward nn with hidden layers"""
    def __init__(self, hidden_nodes, num_inputs=5, num_outputs=2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_nodes = hidden_nodes

        self.w1 = np.zeros((num_inputs, hidden_nodes))
        self.b1 = np.zeros(hidden_nodes)

        self.w2 = np.zeros((hidden_nodes, num_outputs))
        self.b2 = np.zeros(num_outputs)

        self.total_genes = (num_inputs * hidden_nodes) + hidden_nodes + (hidden_nodes * num_outputs) + num_outputs

    def set_weights(self, genome):
        w1_end = self.num_inputs * self.hidden_nodes
        b1_end = w1_end + self.hidden_nodes
        w2_end = b1_end + (self.hidden_nodes * self.num_outputs)

        self.w1 = np.array(genome[:w1_end]).reshape((self.num_inputs, self.hidden_nodes))
        self.b1 = np.array(genome[w1_end:b1_end])
        self.w2 = np.array(genome[b1_end:w2_end]).reshape((self.hidden_nodes, self.num_outputs))
        self.b2 = np.array(genome[w2_end:])

    def forward(self, inputs):
        """
        inputs: [forward, left, right, distance-to-goal, angle-to-goal]
        output: [left_speed, right_speed]
        """
        inputs = np.array(inputs)

        # hidden layer
        hidden = np.dot(inputs, self.w1) + self.b1
        hidden = np.tanh(hidden)

        # output layer
        raw_outputs = np.dot(hidden, self.w2) + self.b2
        left_speed = np.tanh(raw_outputs[0])
        right_speed = np.tanh(raw_outputs[1])

        return left_speed, right_speed
