from drl import train_agent, agent, load_model, PLAYER1_SYMBOL, PLAYER2_SYMBOL, target_agent, ReplayMemory

board_size = 7

player1_pieces = [(0, 0), (2, 0), (4, 6), (6, 6)]
player2_pieces = [(0, 6), (2, 6), (4, 0), (6, 0)]

REPLAY_MEMORY_SIZE = 1000000


def play_game(number_of_move_per_round, max_rounds_per_game, episodes, loaded_model, save_model_as, test, train_option):

    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    board = [[' ' for _ in range(board_size)] for _ in range(board_size)]

    for piece in player1_pieces:
        x, y = piece
        board[x][y] = PLAYER1_SYMBOL

    for piece in player2_pieces:
        x, y = piece
        board[x][y] = PLAYER2_SYMBOL

    if loaded_model != "No Load":
        load_model(agent, loaded_model)

    train_agent(episodes, max_rounds_per_game, number_of_move_per_round, agent, target_agent, memory,
                board, test, save_model_as, train_option)

