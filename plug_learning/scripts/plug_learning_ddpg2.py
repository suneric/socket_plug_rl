#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
from envs.socket_plug_env2 import SocketPlugEnv2
import rospy

"""
Experience buffer
"""
class ExperienceBuffer:
    def __init__(self, buffer_capacity, image_dim, num_actions):
        self.buffer_capacity = buffer_capacity
        self.force_buffer = np.zeros((self.buffer_capacity, 3)) # force x-y-z
        self.position_buffer = np.zeros((self.buffer_capacity, 3)) # joints: 7
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_force_buffer = np.zeros((self.buffer_capacity, 3)) # force x-y-z
        self.next_position_buffer = np.zeros((self.buffer_capacity, 3)) # joints: 7
        self.buffer_counter = 0

    # takes (s,a,r,s') obsercation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        state = obs_tuple[0]
        self.force_buffer[index] = state["force"]
        self.position_buffer[index] = state["position"]

        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]

        next_state = obs_tuple[3]
        self.next_force_buffer[index] = next_state["force"]
        self.next_position_buffer[index] = next_state["position"]
        self.buffer_counter += 1

    # batch sample experiences
    def sample(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)
        # convert to tensors
        force_batch = tf.convert_to_tensor(self.force_buffer[batch_indices])
        position_batch = tf.convert_to_tensor(self.position_buffer[batch_indices])

        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_force_batch = tf.convert_to_tensor(self.next_force_buffer[batch_indices])
        next_position_batch = tf.convert_to_tensor(self.next_position_buffer[batch_indices])
        return dict(
            forces = force_batch,
            positions = position_batch,
            actions = action_batch,
            rewards = reward_batch,
            next_forces = next_force_batch,
            next_positions = next_position_batch
        )

class DDPGAgent:
    def __init__(self,state_dim,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size):
        self.actor_model = self.get_actor(state_dim, num_actions, upper_bound)
        self.critic_model = self.get_critic(state_dim, num_actions)
        self.target_actor = self.get_actor(state_dim, num_actions, upper_bound)
        self.target_critic = self.get_critic(state_dim, num_actions)
        # making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        # experiece buffer
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(buffer_capacity,state_dim,num_actions)
        self.gamma = gamma
        self.tau = tau
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @tf.function
    def update(self, force_batch, position_batch, action_batch, reward_batch, next_force_batch, next_position_batch):
        # training and updating Critic network
        # y_i = r_i + gamma*Q'(s_i+1, u'(s_i+1))
        # crtic loss: L = (1/N)*sum((y_i - Q(s_i,a_i))^2)
        """
        Critic loss - Mean Squared Error of y - Q(s, a) where y is the expected
        return as seen by the Target network, and Q(s, a) is action value predicted
        by the Critic network. y is a moving target that the critic model tries to
        achieve; we make this target stable by updating the Target model slowly.
        """
        with tf.GradientTape() as tape:
            target_actions = self.target_actor([next_force_batch, next_position_batch])
            y = reward_batch + self.gamma * self.target_critic([next_force_batch, next_position_batch, target_actions])
            critic_value = self.critic_model([force_batch, position_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # training and updating Actor network
        """
        Actor loss - This is computed using the mean of the value given by the
        Critic network for the actions taken by the Actor network. We seek to
        maximize this quantity.
        """
        with tf.GradientTape() as tape:
            actions = self.actor_model([force_batch,position_batch])
            critic_value = self.critic_model([force_batch, position_batch, actions])
            # use "-" as we want to maximize the value given by the ctic for our action
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def learn(self):
        experiences = self.buffer.sample(self.batch_size)
        force_batch = experiences['forces']
        position_batch = experiences['positions']
        action_batch = experiences['actions']
        reward_batch = experiences['rewards']
        next_force_batch = experiences['next_forces']
        next_position_batch = experiences['next_positions']
        self.update(force_batch, position_batch, action_batch, reward_batch, next_force_batch, next_position_batch)

    @tf.function
    # Based on rate 'tau', which is much less than one, this update target parameters slowly
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b*self.tau + a*(1-self.tau))

    def get_actor(self, image_dim, num_actions, upper_bound):
        last_init = tf.random_uniform_initializer(minval=-0.003,maxval=0.003)
        #force input
        force_input = tf.keras.layers.Input(shape=(3))
        force_out = tf.keras.layers.Dense(128, activation="relu")(force_input)
        force_out = tf.keras.layers.Dense(64, activation="relu")(force_out)
        # position input
        position_input = tf.keras.layers.Input(shape=(3))
        position_out = tf.keras.layers.Dense(128, activation='relu')(position_input)
        position_out = tf.keras.layers.Dense(64, activation='relu')(position_out)
        # concatenating
        concat = tf.keras.layers.Concatenate()([force_out, position_out])
        out = tf.keras.layers.Dense(64, activation="relu")(concat)
        outputs = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)
        outputs = outputs * upper_bound
        model = tf.keras.Model([force_input, position_input], outputs)
        return model

    def get_critic(self, image_dim, num_actions):
        #force input
        force_input = tf.keras.layers.Input(shape=(3))
        force_out = tf.keras.layers.Dense(128, activation="relu")(force_input)
        force_out = tf.keras.layers.Dense(64, activation="relu")(force_out)
        # position input
        position_input = tf.keras.layers.Input(shape=(3))
        position_out = tf.keras.layers.Dense(128, activation='relu')(position_input)
        position_out = tf.keras.layers.Dense(64, activation='relu')(position_out)
        # action input
        action_input = tf.keras.layers.Input(shape=(num_actions))
        action_out = tf.keras.layers.Dense(64, activation="relu")(action_input)
        # both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([force_out, position_out, action_out])
        out = tf.keras.layers.Dense(64, activation="relu")(concat)
        outputs = tf.keras.layers.Dense(1)(out)
        # output single value for give state-action
        model = tf.keras.Model([force_input, position_input, action_input], outputs)
        return model

    """
    policy returns an action sampled from Actor network plus some noise for exploration
    """
    def policy(self, state):
        force = state['force']
        position = state['position']
        tf_force = tf.expand_dims(force,0)
        tf_position = tf.expand_dims(position,0)
        sampled_actions = tf.squeeze(self.actor_model([tf_force, tf_position]))
        sampled_actions = sampled_actions.numpy()
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        # print(legal_action)
        return legal_action


################################################################################

np.random.seed(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=100)
    parser.add_argument('--max_step', type=int ,default=50)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ddpg_train', anonymous=True, log_level=rospy.INFO)

    critic_lr = 0.002
    actor_lr = 0.001

    maxEpisode = args.max_ep
    maxStep = args.max_step
    gamma = 0.99
    tau = 0.005

    currTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = 'logs/ddpg' + currTime
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = SocketPlugEnv2(resolution=(300,300))
    state_dim = env.state_dimension() # imgae
    print("state dimension",state_dim)
    num_actions = env.action_dimension() # dy, dz
    print("action dimension", num_actions)
    upper_bound = [0.005,0.005]
    lower_bound = [-0.005,-0.005]

    buffer_capacity = 8000
    batch_size = 64

    agent = DDPGAgent(state_dim,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size)

    ep_reward_list = []
    avg_reward_list = []
    for ep in range(maxEpisode):
        state, info = env.reset()
        ep_reward = 0
        for t in range(maxStep):
            action = agent.policy(state)
            new_state, reward, done, _ = env.step(action)
            agent.buffer.record((state,action,reward,new_state))
            ep_reward += reward
            # learn and update target actor and critic network
            agent.learn()
            agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
            agent.update_target(agent.target_critic.variables, agent.critic_model.variables)
            if done:
                break
            state = new_state

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_reward, step=ep)

        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
