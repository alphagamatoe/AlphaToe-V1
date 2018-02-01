"""
Builds and trains a neural network that uses policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board. If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does NOT initially have any way of knowing what is or is not
a valid move, so initially it must learn the rules of the game.

I have trained this version with success at 3x3 tic tac toe until it has a success rate in the region of 75% this maybe
as good as it can do, because 3x3 tic-tac-toe is a theoretical draw, so the random opponent will often get lucky and
force a draw.	
"""
import functools
import collections
import os
import random

import numpy as np
import tensorflow as tf

from network_helpers import create_network,load_network, get_stochastic_network_move, save_network
from tic_tac_toe import TicTacToeGameSpec
from train_policy_gradient import train_policy_gradients

network_file_path = 'network_500000.p'
NUMBER_OF_GAMES_TO_RUN = 2 #Enter the number of games to run

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = TicTacToeGameSpec()

create_network_func = functools.partial(create_network, game_spec.board_squares(), (100, 100, 100))


def input_player_move(board_state, side):
            return game_spec.flat_move_to_tuple(int(input("Next Move:")))
			

reward_placeholder = tf.placeholder("float", shape=(None,))
actual_move_placeholder = tf.placeholder("float", shape=(None, game_spec.outputs()))
input_layer, output_layer, variables = create_network_func()


with tf.Session() as session:    
    if network_file_path and os.path.isfile(network_file_path):
        print("loading pre-existing network")
        load_network(session, variables, network_file_path)
	
    def make_training_move(board_state, side):
            move = get_stochastic_network_move(session, input_layer, output_layer, board_state, side,valid_only = True,game_spec=game_spec)
            return game_spec.flat_move_to_tuple(move.argmax())
	
    for episode_number in range(1, NUMBER_OF_GAMES_TO_RUN):
            # randomize if going first or second
            if bool(random.getrandbits(1)):
                reward,info = game_spec.play_game(make_training_move, input_player_move, log=True)
            else:
                reward,info = game_spec.play_game(input_player_move, make_training_move, log=True)
                reward = -reward
