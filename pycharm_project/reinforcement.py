import random
import string
import time
import numpy as np
import math
import string

import buffers
import layers
import activations
import network


class InvalidNetwork(Exception):
    pass


def mutate_random(values):
    return values + np.random.normal(0, 0.1, size=values.shape)


class ReinforcementTrainer:
    def __init__(self, net, agent_count, top_percentage=0.1):
        self.net = net
        self.temp_network_name = ""

        self.agent_count = agent_count
        self.top_percentage = top_percentage


        # Ready agents
        self.ready_mutation()
        self.agents = [self.create_mutation() for _ in range(self.agent_count)]

    def ready_mutation(self):
        self.temp_network_name = "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".temp"
        self.net.save(self.temp_network_name)

    @staticmethod
    def random_mutate_network(net):
        for layer in net.layout:
            if isinstance(layer, layers.FullyConnectedLayer):
                mutations = mutate_random(layer.weights.get_as_array())
                layer.weights = buffers.NetworkBuffer(mutations, (len(mutations),))

                mutations = mutate_random(layer.biases.get_as_array())
                layer.biases = buffers.NetworkBuffer(mutations, (len(mutations),))

            if isinstance(layer, layers.ConvolutedLayer):
                for i in range(len(layer.weights)):
                    mutations = mutate_random(layer.weights[i].get_as_array())
                    layer.weights[i] = buffers.NetworkBuffer(mutations, (len(mutations),))

                    mutations = mutate_random(layer.biases[i].get_as_array())
                    layer.biases[i] = buffers.NetworkBuffer(mutations, (len(mutations),))

    def create_mutation(self):
        new_network = network.Network.load(self.temp_network_name)
        self.random_mutate_network(new_network)
        return new_network

    def mutate(self, population):
        """
        Generates the next generation by mutating and crossing over the selected population.

        population: List of selected agent IDs (those that passed selection).
        self.agent_count: Total number of agents in the next generation.

        Returns:
        - new_population: List of new agent variations.
        """
        new_population = []

        # Convert population dict keys to list if needed
        population = list(population)

        while len(new_population) < self.agent_count:
            # Select parents randomly from the passing population
            parent1, parent2 = np.random.choice(population, 2, replace=False)

            # Crossover (blend two parents)
            child = self.crossover(parent1, parent2)

            # Apply mutation
            self.random_mutate_network(child)

            # Add to new population
            new_population.append(child)

        return new_population[:self.agent_count]  # Ensure exact size match

    @staticmethod
    def crossover(agent1, agent2, crossover_rate=0.5):
        """ Mixes two parent agents' weights. """
        new_network = network.Network(agent1.layout)

        for layer_index, layer in enumerate(new_network.layout):
            if isinstance(layer, layers.FullyConnectedLayer):
                mutations = np.where(np.random.rand(agent1.layout[layer_index].get_weight_count()) < crossover_rate,
                                     agent1.layout[layer_index].weights.get_as_array(),
                                     agent2.layout[layer_index].weights.get_as_array())
                layer.weights = buffers.NetworkBuffer(mutations, (len(mutations),))

                mutations = np.where(np.random.rand(agent1.layout[layer_index].get_bias_count()) < crossover_rate,
                                     agent1.layout[layer_index].biases.get_as_array(),
                                     agent2.layout[layer_index].biases.get_as_array())
                layer.biases = buffers.NetworkBuffer(mutations, (len(mutations),))

            if isinstance(layer, layers.ConvolutedLayer):
                for i in range(len(layer.weights)):
                    mutations = np.where(np.random.rand(agent1.layout[layer_index].get_weight_count()) < crossover_rate,
                                         agent1.layout[layer_index].weights[i].get_as_array(),
                                         agent2.layout[layer_index].weights[i].get_as_array())
                    layer.weights[i] = buffers.NetworkBuffer(mutations, (len(mutations),))


                    mutations = np.where(np.random.rand(agent1.layout[layer_index].get_bias_count()) < crossover_rate,
                                         agent1.layout[layer_index].biases[i].get_as_array(),
                                         agent2.layout[layer_index].biases[i].get_as_array())
                    layer.biases[i] = buffers.NetworkBuffer(mutations, (len(mutations),))

        return new_network




    def train(self, epoches, training_data):
        for epoch in range(epoches):
            agent_scores = {a: 0 for a in range(self.agent_count)}

            for sample, target in training_data:
                for agent_index, agent in enumerate(self.agents):
                    result = agent.forward_pass(sample)

                    error = sum([min(abs(result[node] - target[node]), 1) for node in range(len(result))])
                    agent_scores[agent_index] += 1 / (error + 1)

            best_score = max(agent_scores.values())
            best_agent = max(agent_scores, key=agent_scores.get)
            self.net = self.agents[best_agent]

            print(f"Best Agent: {best_agent}, Score: {best_score / len(training_data)}")

            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1],
                                   reverse=True)

            top_n = max(1, int(len(sorted_agents) * self.top_percentage))
            top_agents = [self.agents[agent] for agent, score in sorted_agents[:top_n]]

            self.agents = self.mutate(top_agents)



if __name__ == "__main__":
    net = network.Network((
        layers.FullyConnectedLayer(1, 1, activations.ReLU),
    ))

    net.save("test.pyn")

    net = network.Network.load("test.pyn")
    trainer = ReinforcementTrainer(net, 100)

    training_data = [
        [np.array([2]), np.array([4])],
        [np.array([1]), np.array([2])],
        [np.array([3]), np.array([6])],
        [np.array([0]), np.array([0])],
    ]

    trainer.train(100, training_data)

    for point in training_data:
        print(point, net.forward_pass(point[0]))
