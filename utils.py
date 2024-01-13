import datetime

import matplotlib.pyplot as plt

import random

board_size = 7


def move_piece(board, x1, y1, x2, y2, player):
    board[x2][y2] = board[x1][y1]
    board[x1][y1] = ' '
    # print(f"{player} move from ({x1}, {y1}) to ({x2}, {y2})")
    return x1, y1, x2, y2


def capture_after_move(board, player_symbol, opponent_symbol, capture):
    top_wall = [(0, i) for i in range(board_size)]
    bottom_wall = [(board_size - 1, i) for i in range(board_size)]
    left_wall = [(i, 0) for i in range(board_size)]
    right_wall = [(i, board_size - 1) for i in range(board_size)]

    if capture == "capture_opponent":
        i = 1
    elif capture == "capture_own":
        i = 2

    # i = 1 ise player1 , i = 2 ise player2
    if (i == 1):
        opponent_pieces = [(x, y) for x in range(board_size) for y in range(board_size) if
                           board[x][y] == player_symbol]
        player_pieces = [(x, y) for x in range(board_size) for y in range(board_size) if
                         board[x][y] == opponent_symbol]
    else:
        player_pieces = [(x, y) for x in range(board_size) for y in range(board_size) if
                         board[x][y] == player_symbol]
        opponent_pieces = [(x, y) for x in range(board_size) for y in range(board_size) if
                           board[x][y] == opponent_symbol]

    for piece in player_pieces:
        x, y = piece
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # topwall
        capture_conditions_top_wall = [
            ((x + 1, y) in player_pieces and (x + 2, y) in player_pieces and (x + 3, y) in player_pieces and (
                x + 4, y in opponent_pieces)),
            ((x + 1, y) in player_pieces and (x + 2, y) in player_pieces and (x + 3, y) in opponent_pieces),
            ((x + 1, y) in player_pieces and (x + 2, y) in opponent_pieces),
            ((x + 1, y) in opponent_pieces)
        ]

        for condition_top, removal_indices_top in zip(capture_conditions_top_wall,
                                                      ([0, 1, 2, 3], [0, 1, 2], [0, 1], [0])):
            if piece in top_wall and condition_top:
                for idx in removal_indices_top:
                    board[x + idx][y] = ' '
                #print(f"Player {player_symbol} {capture} pieces.")
                return 1
        # bottomwall

        capture_conditions_bottom_wall = [
            ((x - 1, y) in player_pieces and (x - 2, y) in player_pieces and (x - 3, y) in player_pieces and (
                x - 4, y) in opponent_pieces),
            ((x - 1, y) in player_pieces and (x - 2, y) in player_pieces and (x - 3, y) in opponent_pieces),
            ((x - 1, y) in player_pieces and (x - 2, y) in opponent_pieces),
            ((x - 1, y) in opponent_pieces)
        ]

        for condition_bottom, removal_indices_bottom in zip(capture_conditions_bottom_wall,
                                                            ([0, 1, 2, 3], [0, 1, 2], [0, 1], [0])):
            if piece in bottom_wall and condition_bottom:
                for idx in removal_indices_bottom:
                    board[x - idx][y] = ' '
                #print(f"Player {player_symbol} {capture} pieces.")
                return 1

        # leftwall

        capture_conditions_left_wall = [
            ((x, y + 1) in player_pieces and (x, y + 2) in player_pieces and (x, y + 3) in player_pieces and (
                x, y + 4) in opponent_pieces),
            ((x, y + 1) in player_pieces and (x, y + 2) in player_pieces and (x, y + 3) in opponent_pieces),
            ((x, y + 1) in player_pieces and (x, y + 2) in opponent_pieces),
            ((x, y + 1) in opponent_pieces)
        ]

        for condition_left, removal_indices_left in zip(capture_conditions_left_wall,
                                                        ([0, 1, 2, 3], [0, 1, 2], [0, 1], [0])):
            if piece in left_wall and condition_left:
                for idy in removal_indices_left:
                    board[x][y + idy] = ' '
                #print(f"Player {player_symbol} {capture} pieces.")
                return 1

        # rightwall

        capture_conditions_right_wall = [
            ((x, y - 1) in player_pieces and (x, y - 2) in player_pieces and (x, y - 3) in player_pieces and (
                x, y - 4) in opponent_pieces),
            ((x, y - 1) in player_pieces and (x, y - 2) in player_pieces and (x, y - 3) in opponent_pieces),
            ((x, y - 1) in player_pieces and (x, y - 2) in opponent_pieces),
            ((x, y - 1) in opponent_pieces)
        ]

        for condition_right, removal_indices_right in zip(capture_conditions_right_wall,
                                                          ([0, 1, 2, 3], [0, 1, 2], [0, 1], [0])):
            if piece in right_wall and condition_right:
                for idy in removal_indices_right:
                    board[x][y - idy] = ' '
                #print(f"Player {player_symbol} {capture} pieces.")
                return 1

        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if (
                        neighbor1 in opponent_pieces
                        and neighbor2 in opponent_pieces
                        and (x * 2 - neighbor1[0], y * 2 - neighbor1[1]) == neighbor2
                ):
                    for op_piece in opponent_pieces:
                        a, b = op_piece
                        op_neighbors = [(a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)]
                        for op_neighbors1 in op_neighbors:
                            for op_neighbors2 in op_neighbors:
                                if (
                                        op_neighbors1 in player_pieces
                                        and op_neighbors2 in player_pieces
                                        and (a * 2 - op_neighbors1[0], b * 2 - op_neighbors1[1]) == op_neighbors2
                                ):
                                    board[x][y] = ' '
                                    # player +1
                                    board[a][b] = ' '
                                    # player -1
                                    #print(f"Player {player_symbol} {capture} pieces.")
                                    return 1

                    board[x][y] = ' '
                    # player +1
                    #print(f"Player {player_symbol} {capture} pieces.")
                    return 1


def is_valid_move(board, x1, y1, x2, y2, player_symbol):
    if (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1):
        if 0 <= x2 < board_size and 0 <= y2 < board_size:
            if board[x2][y2] == ' ':
                if board[x1][y1] == player_symbol:
                    return True
    return False


def get_valid_moves(board, x1, y1, player_symbol):
    valid_moves = []

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dx, dy in directions:
        x2, y2 = x1 + dx, y1 + dy
        if is_valid_move(board, x1, y1, x2, y2, player_symbol):
            valid_moves.append((x2, y2))

    return valid_moves


def check_game_over(board, step, max_steps_per_episode):  # 0-draw 1-player1 2-player2 9-no over

    player1_count = sum(row.count('▲') for row in board)
    player2_count = sum(row.count('●') for row in board)

    if not any(cell in ['▲', '●'] for row in board for cell in row):
        print("It's a draw! No pieces left on the board.")
        return 0

    if not any(cell == '▲' for row in board for cell in row):
        print("Player 2 wins! Player 1 has no pieces left.")
        return 2

    if not any(cell == '●' for row in board for cell in row):
        print("Player 1 wins! Player 2 has no pieces left.")
        return 1

    if step == max_steps_per_episode - 1:
        if player1_count == player2_count:
            print("It's a draw! Player1 and Player2 has same number of pieces.")
            return 0
        elif player1_count > player2_count:
            print("Player 1 wins! Player 1 has more pieces.")
            return 1
        else:
            print("Player 2 wins! Player 2 has more pieces.")
            return 2

    return 9


def reset_game(board):
    for x in range(board_size):
        for y in range(board_size):
            board[x][y] = ' '

    player1_pieces = [(0, 0), (2, 0), (4, 6), (6, 6)]
    player2_pieces = [(0, 6), (2, 6), (4, 0), (6, 0)]

    for piece in player1_pieces:
        x, y = piece
        board[x][y] = '▲'

    for piece in player2_pieces:
        x, y = piece
        board[x][y] = '●'

    return board


def find_player1_pieces(board):
    player1_pieces = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == '▲':
                player1_pieces.append((row, col))
    return player1_pieces


def print_board(board):
    horizontal_line = "----" * board_size + "---"

    col_numbers = "    " + "   ".join(str(i) for i in range(board_size))
    print(col_numbers)

    for row_num, row in enumerate(board):
        print(horizontal_line)
        print(f"{row_num} |", end="")
        for cell in row:
            print(f" {cell} |", end="")
        print()

    print(horizontal_line)


def player2_random_move(board, number_of_move_per_round):
    player2_pieces = [(x, y) for x in range(board_size) for y in range(board_size) if board[x][y] == '●']

    if not player2_pieces:
        return False

    if len(player2_pieces) == 1:
        number_of_move_per_round = 1

    for _ in range(number_of_move_per_round):
        piece_to_move = random.choice(player2_pieces)
        x1, y1 = piece_to_move
        valid_moves = get_valid_moves(board, x1, y1, '●')

        while not valid_moves:
            player2_pieces.remove(piece_to_move)
            if not player2_pieces:
                return False
            piece_to_move = random.choice(player2_pieces)
            x1, y1 = piece_to_move
            valid_moves = get_valid_moves(board, x1, y1, '●')

        move = random.choice(valid_moves)
        x2, y2 = move
        move_piece(board, x1, y1, x2, y2, "player2")
        capture_after_move(board, '●', '▲', "capture_opponent")
        capture_after_move(board, '●', '▲', "capture_own")

    return True


def plot_win_rate(win_rates):
    ct = datetime.datetime.now()
    save_name = "plots/win-rate-plot-" + ct.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    games = range(0, len(win_rates) * 10, 10)
    plt.plot(games, win_rates)
    plt.title('Player 1 Win Rate per Games')
    plt.xlabel('Games')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)

    if win_rates:
        last_game = games[-1]
        last_win_rate = win_rates[-1]
        plt.text(last_game, last_win_rate, f'{last_win_rate}%', ha='right', va='bottom')

    plt.savefig(save_name, dpi=300)
    plt.show()


def plot_win_loss(win, loss, draw):
    ct = datetime.datetime.now()
    save_name = "plots/end-game-count-plot-" + ct.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    games = [i * 10 for i in range(max(len(win), len(loss), len(draw)))]
    fig, ax = plt.subplots()

    ax.plot(games, win, color='green', linestyle='-', label='Wins')
    ax.plot(games, loss, color='red', linestyle='-.', label='Loss')
    ax.plot(games, draw, color='gray', linestyle='--', label='Draws')

    ax.set_xlabel('Games')
    ax.set_ylabel('Count')
    ax.tick_params(axis='y')
    ax.grid(True)
    plt.title('Player 1 Wins, Losses, and Draws per Games')

    if win:
        ax.text(games[-1], win[-1], f'{win[-1]}', ha='right', va='bottom')
    if loss:
        ax.text(games[-1], loss[-1], f'{loss[-1]}', ha='right', va='bottom')
    if draw:
        ax.text(games[-1], draw[-1], f'{draw[-1]}', ha='right', va='bottom')

    ax.legend(loc='best')

    plt.savefig(save_name, dpi=300)
    plt.show()


def plot_reward(total_rewards):
    ct = datetime.datetime.now()
    save_name = "plots/total-reward-plot-" + ct.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
    episodes = range(1, len(total_rewards) + 1)
    plt.plot(episodes, total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid(True)

    if total_rewards:
        last_episode = episodes[-1]
        last_total_reward = total_rewards[-1]
        plt.text(last_episode, last_total_reward, f'{last_total_reward}', ha='right', va='bottom')

    plt.savefig(save_name, dpi=300)
    plt.show()


def end_game_data(player1_win_count, player2_win_count, draw_count, win_rates, player1_wins, player2_wins, draws,
                  loss_values, total_rewards):
    total_games = (player1_win_count + player2_win_count + draw_count)
    player1_win_rate = player1_win_count / total_games * 100
    player2_win_rate = player2_win_count / total_games * 100
    draw_rate = draw_count / total_games * 100

    print(f"Player 1 Win Rate: %{player1_win_rate:.2f}")
    print(f"Player 2 Win Rate: %{player2_win_rate:.2f}")
    print(f"Draw Rate: %{draw_rate:.2f}")

    print(win_rates)
    print(player1_wins)
    print(player2_wins)
    print(draws)
    print(total_rewards)

    plot_win_rate(win_rates)
    plot_win_loss(player1_wins, player2_wins, draws)
    plot_reward(total_rewards)

    if loss_values:
        print("Final Average Loss:", sum(loss_values) / len(loss_values))

    return player1_win_rate
