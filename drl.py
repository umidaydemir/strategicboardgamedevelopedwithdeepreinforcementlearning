import os
from collections import namedtuple, deque
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import is_valid_move, capture_after_move, move_piece, board_size, check_game_over, reset_game, \
    get_valid_moves, end_game_data, player2_random_move
from events import win_rate_updated_event

PLAYER1_SYMBOL = '▲'
PLAYER2_SYMBOL = '●'

FALL_INTO_TO_TRAP = -5
TRAP_TO_CAPTURE = 3
MAKE_CHANCE_TO_CAPTURE = 3
GIVE_CHANCE_TO_CAPTURE = -5
CORNER_REWARD = -2
EDGE_REWARD = -1
CAPTURE_OWN_REWARD = -6
CAPTURE_OPPONENT_REWARD = 6
WIN_VALUE = 10
DRAW_VALUE = 0.0
LOSS_VALUE = -10

BOARD_SIZE = 7

INPUT_SIZE = 49
HIDDEN_SIZE = 196
OUTPUT_SIZE = 45

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.001
TARGET_UPDATE = 10

GAMMA = 0.99

Experience = namedtuple('Experience', ('state', 'action', 'new_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, new_state, reward):
        self.memory.append(Experience(state, action, new_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()

        self.dl1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.dl2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.output_layer = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        return x


agent = DQNAgent()
target_agent = DQNAgent()
target_agent.load_state_dict(agent.state_dict())

optimizer = optim.AdamW(agent.parameters(), lr=LEARNING_RATE, amsgrad=True)


def board_to_state(board):
    state = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == PLAYER1_SYMBOL:
                state[x][y] = 1
            elif board[x][y] == PLAYER2_SYMBOL:
                state[x][y] = -1
    return state.flatten()


def select_action(state, dqn_agent, output_size, epsilon):
    if random.random() < epsilon:
        return random.randint(0, output_size - 1)
    else:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs = dqn_agent(state_tensor)
        return torch.argmax(action_probs).item()


def decode_action(action, board):
    piece_index = action // 10 % 4
    move_index = action % 10 % 4

    piece_coords = find_piece_coordinates(piece_index, board)
    move_coords = select_valid_move(board, piece_coords, PLAYER1_SYMBOL, move_index)

    x1, y1 = piece_coords
    x2, y2 = move_coords

    return x1, y1, x2, y2


def select_valid_move(board, piece, player_symbol, move_index):
    x, y = piece
    valid_moves = get_valid_moves(board, x, y, player_symbol)
    if move_index < len(valid_moves):
        return valid_moves[move_index]
    else:
        return valid_moves[0] if valid_moves else (0, 0)


def find_piece_coordinates(piece_index, board):
    player1_pieces = [(x, y) for x in range(len(board)) for y in range(len(board)) if board[x][y] == PLAYER1_SYMBOL]

    if piece_index < len(player1_pieces):
        return player1_pieces[piece_index]
    else:
        return player1_pieces[0] if player1_pieces else (0, 0)


def punish_edge(x2, y2):
    corners = [(0, 0), (0, 6), (6, 0), (6, 6)]

    if (x2, y2) in corners:
        return CORNER_REWARD
    else:
        if x2 == 0 or x2 == 6 or y2 == 0 or y2 == 6:
            return EDGE_REWARD
        else:
            return 0


def check_surrounding(board, x, y):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_dirs = {(-1, 0): [(1, -1), (1, 1)],
                     (1, 0): [(-1, -1), (-1, 1)],
                     (0, -1): [(1, 1), (-1, 1)],
                     (0, 1): [(1, -1), (-1, -1)]}

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx][ny] == PLAYER2_SYMBOL:
            for ddx, ddy in diagonal_dirs[(dx, dy)]:
                dx2, dy2 = x + ddx, y + ddy
                if 0 <= dx2 < BOARD_SIZE and 0 <= dy2 < BOARD_SIZE and board[dx2][dy2] == PLAYER2_SYMBOL:
                    return GIVE_CHANCE_TO_CAPTURE
    return 0


def check_combined_surrounding(board, x, y):
    straight_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_dirs = {(-1, 0): [(-1, -1), (-1, 1)],
                     (1, 0): [(1, -1), (1, 1)],
                     (0, -1): [(-1, -1), (1, -1)],
                     (0, 1): [(-1, 1), (1, 1)]}

    for dx, dy in straight_dirs:
        adjacent_x, adjacent_y = x + dx, y + dy
        if 0 <= adjacent_x < BOARD_SIZE and 0 <= adjacent_y < BOARD_SIZE and board[adjacent_x][
            adjacent_y] == PLAYER2_SYMBOL:
            for diag_dx, diag_dy in diagonal_dirs[(dx, dy)]:
                diag_x, diag_y = adjacent_x + diag_dx, adjacent_y + diag_dy
                if 0 <= diag_x < BOARD_SIZE and 0 <= diag_y < BOARD_SIZE and board[diag_x][diag_y] == PLAYER1_SYMBOL:
                    return MAKE_CHANCE_TO_CAPTURE

    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for ddx, ddy in diagonal_dirs:
        diag_x, diag_y = x + ddx, y + ddy
        if 0 <= diag_x < BOARD_SIZE and 0 <= diag_y < BOARD_SIZE and board[diag_x][diag_y] == PLAYER2_SYMBOL:
            side_x, side_y = diag_x + ddx, diag_y
            if 0 <= side_x < BOARD_SIZE and 0 <= side_y < BOARD_SIZE and board[side_x][side_y] == PLAYER1_SYMBOL:
                return MAKE_CHANCE_TO_CAPTURE

    return 0


def check_surrounding_for_capture(board, x, y):
    # Kontrol edilecek yönler
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:

        space_x, space_y = x + dx, y + dy
        opponent_x, opponent_y = x + dx * 2, y + dy * 2
        wall_x, wall_y = x + dx * 3, y + dy * 3

        if 0 <= space_x < BOARD_SIZE and 0 <= space_y < BOARD_SIZE and board[space_x][space_y] == ' ':
            if 0 <= opponent_x < board_size and 0 <= opponent_y < BOARD_SIZE and board[opponent_x][
                opponent_y] == PLAYER2_SYMBOL:
                if wall_x < 0 or wall_x >= BOARD_SIZE or wall_y < 0 or wall_y >= BOARD_SIZE:
                    return TRAP_TO_CAPTURE
                elif 0 <= wall_x < BOARD_SIZE and 0 <= wall_y < BOARD_SIZE and board[wall_x][
                    wall_y] == PLAYER2_SYMBOL:
                    wall_x, wall_y = x + dx * 4, y + dy * 4
                    if wall_x < 0 or wall_x >= BOARD_SIZE or wall_y < 0 or wall_y >= BOARD_SIZE:
                        return TRAP_TO_CAPTURE * 2

    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in diagonal_dirs:
        opponent_x, opponent_y = x + dx, y + dy
        wall_x, wall_y = x + dx * 2, x + dy * 2

        if 0 <= opponent_x < board_size and 0 <= opponent_y < BOARD_SIZE and board[opponent_x][
            opponent_y] == PLAYER2_SYMBOL:
            if wall_x < 0 or wall_x >= BOARD_SIZE or wall_y < 0 or wall_y >= BOARD_SIZE:
                return TRAP_TO_CAPTURE

    return 0


def trap_itself(board, x, y):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        wall_x, wall_y = x - dx, y - dy

        if 0 <= wall_x < BOARD_SIZE and 0 <= wall_y < BOARD_SIZE and board[wall_x][wall_y] == PLAYER1_SYMBOL:
            wall_x, wall_y = x - dx * 2, y - dy * 2

        if wall_x < 0 or wall_x >= BOARD_SIZE or wall_y < 0 or wall_y >= BOARD_SIZE:
            next_x, next_y = x + dx, y + dy
            if 0 <= next_x < BOARD_SIZE and 0 <= next_y < BOARD_SIZE and board[next_x][next_y] == ' ':
                opponent_x, opponent_y = x + dx * 2, y + dy * 2
                if 0 <= opponent_x < BOARD_SIZE and 0 <= opponent_y < BOARD_SIZE and board[opponent_x][
                    opponent_y] == PLAYER2_SYMBOL:
                    return FALL_INTO_TO_TRAP
            elif 0 <= next_x < BOARD_SIZE and 0 <= next_y < BOARD_SIZE and board[next_x][next_y] == PLAYER1_SYMBOL:
                space_x, space_y = x + dx * 2, y + dy * 2
                if 0 <= space_x < BOARD_SIZE and 0 <= space_y < BOARD_SIZE and board[space_x][space_y] == ' ':
                    opponent_x, opponent_y = x + dx * 3, y + dy * 3
                    if 0 <= opponent_x < BOARD_SIZE and 0 <= opponent_y < BOARD_SIZE and board[opponent_x][
                        opponent_y] == PLAYER2_SYMBOL:
                        return FALL_INTO_TO_TRAP * 2

    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in diagonal_dirs:
        opponent_x, opponent_y = x + dx, y + dy
        wall_x, wall_y = x - dx, x - dy

        if 0 <= opponent_x < board_size and 0 <= opponent_y < BOARD_SIZE and board[opponent_x][
            opponent_y] == PLAYER2_SYMBOL:
            if wall_x < 0 or wall_x >= BOARD_SIZE or wall_y < 0 or wall_y >= BOARD_SIZE:
                return FALL_INTO_TO_TRAP

    return 0


def apply_action_to_game(action, board, state, current_step, max_steps_per_episode):
    new_state = state
    reward = 0
    done = check_game_over(board, current_step, max_steps_per_episode)

    if done != 9:
        return state, reward, done

    x1, y1, x2, y2 = decode_action(action, board)

    if is_valid_move(board, x1, y1, x2, y2, PLAYER1_SYMBOL):
        move_piece(board, x1, y1, x2, y2, "player1")
        new_state = board_to_state(board)
        if capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_opponent") == 1:
            reward += CAPTURE_OPPONENT_REWARD
        elif capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_own") == 1:
            reward += CAPTURE_OWN_REWARD

        reward += punish_edge(x2, y2)
        reward += check_surrounding(board, x2, y2)
        reward += check_combined_surrounding(board, x2, y2)
        reward += check_surrounding_for_capture(board, x2, y2)
        reward += trap_itself(board, x2, y2)

    done = check_game_over(board, current_step, max_steps_per_episode)

    if done == 0:
        reward += DRAW_VALUE
        new_state = board_to_state(board)
    elif done == 1:
        reward += WIN_VALUE
        new_state = board_to_state(board)
    elif done == 2:
        reward = reward + LOSS_VALUE
        new_state = board_to_state(board)

    return new_state, reward, done


def play_round(number_of_move_per_round, step, board, policy_net, epsilon, memory, max_steps_per_episode):
    player1_piece_coords = [(x, y) for x in range(board_size) for y in range(board_size) if
                            board[x][y] == PLAYER1_SYMBOL]

    if len(player1_piece_coords) == 1:
        number_of_move_per_round = 1

    for _ in range(number_of_move_per_round):
        state = board_to_state(board)
        action = select_action(state, policy_net, OUTPUT_SIZE, epsilon)

        new_state, reward, done = apply_action_to_game(action, board, state, step, max_steps_per_episode)

        memory.push(state, action, new_state, reward)

        if done != 9:
            break

    return board, done, reward


def learn(memory, batch_size, policy_net, target_net, loss_values):
    if len(memory) > batch_size:
        experiences = memory.sample(batch_size)
        states, actions, new_states, rewards = zip(*experiences)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        new_states = torch.FloatTensor(np.array(new_states))
        rewards = torch.FloatTensor(np.array(rewards))

        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))

        max_next_q_values = target_net(new_states).max(1)[0].detach()
        expected_q_values = rewards + (GAMMA * max_next_q_values)

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

    return loss_values


def train_agent(episodes, max_steps_per_episode, number_of_move_per_round, policy_net, target_net, memory, board,
                test, save_model_as, train_option):
    total_rewards = [0]
    player1_wins = [0]
    player2_wins = [0]
    draws = [0]
    win_rates = [0]
    loss_values = []
    player1_win_rate = 0
    player1_win_count = 0
    player2_win_count = 0
    draw_count = 0

    for episode in range(episodes):
        print(f"GAME {episode + 1}")
        if test:
            epsilon = 0
        else:
            if train_option == "Random vs Random":
                epsilon = 1
            else:
                epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_DECAY * episode))
                # epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))
                # epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** (episode*2)))
                # epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** (episode * 2)))

        total_reward = 0

        for step in range(max_steps_per_episode):
            board, done, reward = play_round(number_of_move_per_round, step, board, policy_net, epsilon, memory,
                                             max_steps_per_episode)
            loss_values = learn(memory, BATCH_SIZE, policy_net, target_net, loss_values)
            total_reward += reward

            if done == 1:
                player1_win_count += 1
                print(f"Player 1 Win Count: {player1_win_count}")
                break
            elif done == 2:
                player2_win_count += 1
                print(f"Player 2 Win Count: {player2_win_count}")
                break
            elif done == 0:
                draw_count += 1
                print(f"Draw Count: {player2_win_count}")
                break

            elif done == 9:
                if train_option == "Agent vs Random":
                    player2_random_move(board, number_of_move_per_round)
                elif train_option == "Agent vs Agent":
                    player2_trained_agent_move(board, policy_net)
                elif train_option == "Random vs Random":
                    player2_random_move(board, number_of_move_per_round)

        if (episode + 1) % TARGET_UPDATE == 0:
            target_agent.load_state_dict(agent.state_dict())
            if len(loss_values) >= 10:
                average_loss = sum(loss_values[-10:]) / 10
                print(f"Episode {episode}: Average Loss = {average_loss}")
            else:
                print(f"Episode {episode}: Insufficient data for average loss calculation.")

            total_games = (player1_win_count + player2_win_count + draw_count)

            if total_games > 0:
                player1_win_rate = player1_win_count / total_games * 100
                player2_win_rate = player2_win_count / total_games * 100
                draw_rate = draw_count / total_games * 100

            win_rates.append(player1_win_rate)
            player1_wins.append(player1_win_count)
            player2_wins.append(player2_win_count)
            draws.append(draw_count)

        print(f"total_reward:{total_reward}")
        total_rewards.append(total_reward)
        board = reset_game(board)

    player1_win_rate = end_game_data(player1_win_count, player2_win_count, draw_count, win_rates, player1_wins,
                                     player2_wins, draws, loss_values, total_rewards)

    win_rate_updated_event.emit(player1_win_rate, player2_win_rate, draw_rate)

    if not test:
        if save_model_as != "":
            save_model(target_agent, save_model_as)


def save_model(model, filename):
    path = "models/" + filename + ".pth"
    try:
        torch.save(model.state_dict(), path)
        print(f"Model successfully saved to {path}.")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model, filename):
    path = "models/" + filename
    if os.path.isfile(path):
        try:
            print("Model is succesfully loaded.")
            model.load_state_dict(torch.load(path))
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"No model file found at {path}.")


def player2_trained_agent_move(board, trained_agent):
    for i in range(7):
        for j in range(7):
            if board[i][j] == PLAYER1_SYMBOL:
                board[i][j] = PLAYER2_SYMBOL
            elif board[i][j] == PLAYER2_SYMBOL:
                board[i][j] = PLAYER1_SYMBOL

    state = board_to_state(board)

    action = select_action(state, trained_agent, 45, 0)

    x1, y1, x2, y2 = decode_action(action, board)

    for i in range(7):
        for j in range(7):
            if board[i][j] == PLAYER1_SYMBOL:
                board[i][j] = PLAYER2_SYMBOL
            elif board[i][j] == PLAYER2_SYMBOL:
                board[i][j] = PLAYER1_SYMBOL

    if is_valid_move(board, x1, y1, x2, y2, PLAYER2_SYMBOL):
        move_piece(board, x1, y1, x2, y2, "player2")
        capture_after_move(board, PLAYER2_SYMBOL, PLAYER1_SYMBOL, "capture_opponent")
        capture_after_move(board, PLAYER2_SYMBOL, PLAYER1_SYMBOL, "capture_own")

    return board, x1, y1, x2, y2
