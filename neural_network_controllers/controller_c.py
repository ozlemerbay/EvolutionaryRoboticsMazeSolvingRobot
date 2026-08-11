import numpy as np

class ControllerC:
    """Elman-style recurrent nn with one hidden layer"""
    def __init__(self, hidden_nodes, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_nodes = hidden_nodes

        self.w_in = np.zeros((num_inputs, hidden_nodes))
        self.w_h = np.zeros((hidden_nodes, hidden_nodes))
        self.b_h = np.zeros(hidden_nodes)

        self.w_out = np.zeros((hidden_nodes, num_outputs))
        self.b_out = np.zeros(num_outputs)

        # where memory is kept
        self.hidden_state = np.zeros(hidden_nodes)

        self.genome_len = (
            num_inputs * hidden_nodes
            + hidden_nodes * hidden_nodes
            + hidden_nodes
            + hidden_nodes * num_outputs
            + num_outputs
        )

    def set_weights(self, genome):
        idx = 0

        w_in_end = idx + self.num_inputs * self.hidden_nodes
        self.w_in = np.array(genome[idx:w_in_end]).reshape((self.num_inputs, self.hidden_nodes))
        idx = w_in_end

        w_h_end = idx + self.hidden_nodes * self.hidden_nodes
        self.w_h = np.array(genome[idx:w_h_end]).reshape((self.hidden_nodes, self.hidden_nodes))
        idx = w_h_end

        b_h_end = idx + self.hidden_nodes
        self.b_h = np.array(genome[idx:b_h_end])
        idx = b_h_end

        w_out_end = idx + self.hidden_nodes * self.num_outputs
        self.w_out = np.array(genome[idx:w_out_end]).reshape((self.hidden_nodes, self.num_outputs))
        idx = w_out_end
        self.b_out = np.array(genome[idx:])

        self.hidden_state = np.zeros(self.hidden_nodes)

    def forward(self, inputs):
        """
        inputs: [forward, left, right, distance-to-goal, angle-to-goal]
        output: [left_speed, right_speed]
        """
        x = np.array(inputs)

        # merge input layer values with the hidden state from previous calculation to create a memory feature
        linear_combination = np.dot(x, self.w_in) + np.dot(self.hidden_state, self.w_h) + self.b_h
        self.hidden_state = np.tanh(linear_combination)

        # calculate the output using new hidden state (updated memory)
        raw_outputs = np.dot(self.hidden_state, self.w_out) + self.b_out

        left_speed = np.tanh(raw_outputs[0])
        right_speed = np.tanh(raw_outputs[1])

        return left_speed, right_speed
