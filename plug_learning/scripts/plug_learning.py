#!/usr/bin/env python3
import sys
sys.path.append('..')
sys.path.append('.')
import os
import numpy as np
import time
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
from envs.socket_plug_env import SocketPlugEnv
from agents.ppo import PPOAgent, ReplayBuffer
import rospy
import tensorflow as tf
import argparse

# application wise random seed
np.random.seed(123)

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--max_ep', type=int, default=10000)
    parser.add_argument('--max_step', type=int, default=50)
    return parser.parse_args()

"""
PPO training with visual observation and forces information fusion
"""
if __name__=='__main__':
    args = get_args()
    print("camera noise", args.noise, "max episode", args.max_ep, "max step", args.max_step)

    buffer_cap = 600 # buffer capacity
    train_freq = 500 # training when number of experiences > 500
    # hyper parameters
    gamma = 0.99 # discount rate
    lamda = 0.97 #
    beta = 0.001 # influence of entropy loss
    clip_ratio = 0.2 # clip_ratio
    actor_lr = 1e-4 # learning rate of actor
    critic_lr = 3e-4 # learning rate actor
    iter_a = 80 # training iteration of actor net
    iter_c = 80 # training iteration of critic net

    rospy.init_node('ppo_train', anonymous=True, log_level=rospy.INFO)
    rospy.sleep(1) #

    # statistics record
    model_dir = os.path.join(sys.path[0], '..', 'saved_models', 'plug', 'noise'+str(args.noise), datetime.now().strftime("%Y-%m-%d-%H-%M"))
    print("model is saved to", model_dir)
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()

    #
    env = SocketPlugEnv(resolution=(64,64), cam_noise=args.noise)
    action_size = env.action_dimension()
    visual_dim = env.visual_dimension()
    agent = PPOAgent(image_dim=visual_dim, force_dim=3, action_size=action_size, clip_ratio=clip_ratio, lr_a=actor_lr, lr_c=critic_lr, beta=beta)
    buffer = ReplayBuffer(image_shape=visual_dim, force_shape=3, action_size=action_size, size=buffer_cap)

    start_time = time.time()
    success_counter = 0
    for ep in range(args.max_ep):
        # run trajectory
        obs, info = env.reset()
        ep_ret, ep_len = 0, 0
        for st in range(args.max_step):
            pred, act, val = agent.action(obs)
            n_obs, rew, done, info = env.step(act)
            buffer.store(obs, tf.one_hot(act,action_size).numpy(), rew, pred, val)
            obs = n_obs
            ep_ret += rew
            ep_len += 1
            # print(info)
            if done:
                break;

        # log info
        tf.summary.scalar("episode total reward", ep_ret, step=ep+1)
        if env.success:
            success_counter += 1
        rospy.loginfo(
            "\n----\nEpisode: {}, EpReturn: {}, EpLength: {}, Succeeded: {}\n----\n".format(
                ep+1,
                ep_ret,
                ep_len,
                success_counter
            )
        )

        # update the total reward, advantages once each episode is completed
        buffer.ep_update(gamma=gamma, lamda=lamda)
        # train models every 500 experiences
        size = buffer.size()
        if size >= train_freq or (ep+1) == args.max_ep:
            print("ppo training with ",size," experiences...")
            agent.train(data=buffer.get(), batch_size=size, iter_a=iter_a, iter_c=iter_c)

        # save models every 500 episodes
        if not (ep+1)%500:
            logits_net_path = os.path.join(model_dir, 'logits_net', str(ep+1))
            val_net_path = os.path.join(model_dir, 'val_net', str(ep+1))
            agent.save(logits_net_path, val_net_path)
            print("save weights to ", model_dir)
