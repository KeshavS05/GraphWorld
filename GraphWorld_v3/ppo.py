import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import scipy.signal
import time

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        # 2D np arrays
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        # 1D np arrays
        num_agents = 2
        self.action_buffer = np.zeros((size, num_agents), dtype=np.int32)

        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros((size, num_agents), dtype=np.float32)
       
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = tf.keras.Dense(units=size, activation=activation)(x)
    return tf.keras.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    # logprobabilities_all = tf.nn.log_softmax(logits)
    # logprobability = tf.reduce_sum(
    #     tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    # )
    # return logprobability
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    num_actions = 5
    # logprobabilities_all = tf.nn.log_softmax(logits)
    # one_hot_actions = tf.one_hot(a, num_actions)
    # logprobability = tf.reduce_sum(
    #     one_hot_actions * logprobabilities_all, axis=-1
    # )
    # return logprobability
    
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobabilities_all = tf.reshape(logprobabilities_all, [-1, num_agents, num_actions])  # reshape to [batch_size, num_agents, num_actions]

    actions = tf.reshape(a, [-1, num_agents])  # reshape actions to [batch_size, num_agents]
    one_hot_actions = tf.one_hot(actions, num_actions)  # one-hot encode separately for each agent

    logprobability = tf.reduce_sum(one_hot_actions * logprobabilities_all, axis=-1)
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation, actor):
    print("???????????????????????????????")
    # logits = actor(observation)
    # action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    num_agents = 2
    num_actions = 5
    logits = actor(observation)  # shape: (batch_size, num_agents * num_actions)
    logits = tf.reshape(logits, (-1, num_agents, num_actions))  # shape: (batch_size, num_agents, num_actions)
    actions = []
    for i in range(num_agents):
        action_i = tf.squeeze(tf.random.categorical(logits[:, i, :], 1), axis=-1)  # shape: (batch_size,)
        actions.append(action_i)
    actions = tf.stack(actions, axis=-1)  # shape: (batch_size, num_agents)
   
    
    print(f"Logits{logits} action:{actions}")
    return logits, actions


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, actor, policy_optimizer, num_actions, clip_ratio=0.2
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer, num_actions)
            - logprobability_buffer
        )
        # Expand dimensions of the advantage_buffer to match the dimensions of ratio
        advantage_buffer = tf.expand_dims(advantage_buffer, axis=-1)


        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer, num_actions)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer, critic, value_optimizer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    
def normalize(observation):
    # Normalize the observation using Min-Max normalization
    min_value = np.min(observation)
    max_value = np.max(observation)
    normalized_observation = (observation - min_value) / (max_value - min_value)
    return normalized_observation
   
    
    
    
def training_loop(env,  actor, critic, buffer, policy_optimizer, value_optimizer):
    print(f"------------------Training Started------------------")
    # Hyperparameters of the PPO algorithm
    # steps_per_epoch = 4000
    steps_per_epoch = 20
    epochs = 1
    gamma = 0.99
    clip_ratio = 0.2

    policy_learning_rate = 3e-4
    value_function_learning_rate = 1e-3

    train_policy_iterations = 80
    train_value_iterations = 80

    lam = 0.97
    target_kl = 0.01
    hidden_sizes = (64, 64)
    render = True


    # Initialize the observation, episode return and episode length
    observation, episode_return, episode_length = env.reset(), 0, 0



    # Iterate over the number of epochs
    epochs = 10
    steps_per_epoch = 30
    tota_sum_return = []
    total_rewards_list_each_epoch = []
    total_kl_list_each_epoch = []
    for epoch in range(epochs):
        print(f"------------------Epoch {epoch}------------------")
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        rewards_list_each_step_in_epoch = []

        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):
            print(f"""------------------Step {t}------------------""")
            # if render:
            #      env.render()
#
            # Get the logits, action, and take one step in the environment
            #observation = observation.reshape(1, -1)
            observation = np.array(observation).reshape(1, -1)
            observation = observation[0]
            observation = np.array(observation).reshape(1, -1)
            #print(f"Observation{observation}")
            # observation = normalize(observation)
            # print(f"Normalized Observation{observation}")
            
            logits, actions = sample_action(observation, actor)
            
            print(f"Observation{observation} Action:{actions} logits:{logits}")
            
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            action = env.take_action(actions[0])
            print("new action", action)
            observation_new, reward, done, _ = env.step(action)
            # observation_new = normalize(observation_new)
            # print(f"Normalized Observation{observation_new}")
            print(f"Observation new{observation_new} reward:{reward} done:{done}")
            rewards_list_each_step_in_epoch.append(reward)
            
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            print(f"Value{value_t}")
            logprobability_t = logprobabilities(logits, actions, num_actions)
            print(f"logprobability_t{logprobability_t}")

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, actions, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                observation = np.array(observation).reshape(1, -1)
                observation = observation[0]
                observation = np.array(observation).reshape(1, -1)
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()
        print("------------------Training Policy Started------------------")
        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, actor, policy_optimizer, num_actions, clip_ratio=0.2
            )
            print(f":::::::::::::::::::KL{kl} for target Kl {target_kl}:::::::::::::::::::::::")
            total_kl_list_each_epoch.append(kl)
            # if kl > 1.5 * target_kl: #original
            #     print("Early stopping at step {}".format(_))
            #     # Early Stopping
            #     break
            
            if kl > 2.25: #original
                print("Early stopping at step {}".format(_))
                # Early Stopping
                break
        print("------------------Training Value Started------------------")
        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer, critic, value_optimizer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
        tota_sum_return.append(sum_return / num_episodes)
        total_rewards_list_each_epoch.append(rewards_list_each_step_in_epoch)
        
    print(f"total sum return {tota_sum_return}")
    print(f"total rewards list each epoch {total_rewards_list_each_epoch}")
    print(f"total kl list each epoch {total_kl_list_each_epoch}")
    
    
    
        
def testing_loop(env, actor, render=False):
    print(f"------------------Testing Started------------------")
    # Define the number of testing episodes
    num_test_episodes = 10

    # Initialize a list to store the test returns
    test_returns = []

    for _ in range(num_test_episodes):
        # Initialize the observation, episode return, and episode length
        observation, episode_return, episode_length = env.reset(), 0, 0

        while True:
            if render:
                env.render()

            # # Sample action from actor
            # observation = observation.reshape(1, -1)
            # logits, action = sample_action(observation, actor)

            # # Take action in the environment
            # observation, reward, done, _ = env.step(action[0].numpy())
            
            #observation = np.array(observation).reshape(1, -1)
            observation = observation[0]
            observation = np.array(observation).reshape(1, -1)
            print(f"Observation{observation}")
            logits, actions = sample_action(observation, actor)
            
            print(f"Observation{observation} Action:{actions} logits:{logits}")
            
            
            action = env.take_action(actions[0])
            print("new action", action)
            observation_new, reward, done, _ = env.step(action)
            print(f"Observation{observation_new} reward:{reward} done:{done}")
            episode_return += reward
            episode_length += 1

            # If terminal state reached, finish the episode
            if done:
                break

        # Append the return for this episode
        test_returns.append(episode_return)

    # Print mean return in test episodes
    print(f"Test returns: {test_returns}")
    print(f"Mean return in test: {np.mean(test_returns)}")
    


 # True if you want to render the environment
render = False

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions

# env = gym.make("CartPole-v0", render_mode='human')
# observation_dimensions = env.observation_space.shape[0]
# num_actions = env.action_space.n
# print("+------------------------+")
# print(f"Observation space: {env.observation_space}\n Actions space: {env.action_space}")
# print(f"\nObservation dimensions: {observation_dimensions}\nNumber of actions: {num_actions}")
# print("+------------------------+") 

import networkx as nx
from multi_agents_graph import MultiAgentGraphWorldEnv
from multi_agents_graph import Graph
import numpy as np
### Initialize GraphEnvironment
n_agents = 2
nodes = [0, 1, 2, 3, 4]
edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
graph1 = nx.Graph()
graph1.add_nodes_from(nodes)
graph1.add_edges_from(edges)
# Generate adjacency matrix

graph = Graph()
graph.graph = graph1
adjacency_matrix = nx.adjacency_matrix(graph1)
env = MultiAgentGraphWorldEnv(graph1, n_agents)
env.env_graph_adjM_original = graph.env_graph_adjM_original()
env.env_graph_adjM_with_edge_cost = graph.env_graph_adjM_with_edge_cost()
env.env_graph_adjM_with_reduced_cost = graph.env

env.observation_space = np.array(env.get_observation_space())
observation_dimensions = env.observation_space.shape[0]

env.action_space = env.get_action_space()


num_actions = 10
print(f"Observation dimensions: {env.observation_space.shape}")
print(f"Observation dimensions: {observation_dimensions}")
# print(f"Action space: {env.action_space}")
print(num_actions)




### Initialize hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2

policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3

train_policy_iterations = 80
train_value_iterations = 80

lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)
# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# # Initialize the actor and the critic as keras models
# observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)

# # actor part
# logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
# actor = keras.Model(inputs=observation_input, outputs=logits)

# Initialize the actor and the critic as keras models
num_actions = 5
num_agents = 2
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)

# actor part
logits = mlp(observation_input, list(hidden_sizes) + [num_agents * num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)

# critic part
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)



# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
print(f"--------------------Model summary---------------------")

print(f"Actor summary: {actor.summary()}")
print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print(f"Critic summary: {critic.summary()}")

print("---------------Start training-----------------")
training_loop(env, actor, critic, buffer, policy_optimizer, value_optimizer)
#testing_loop(env, actor)

