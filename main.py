from utils import capture_after_move, is_valid_move, move_piece

from drl import select_action, board_to_state, decode_action, PLAYER1_SYMBOL, PLAYER2_SYMBOL, load_model, DQNAgent


def ai_move(board, loaded_model):
    trained_agent = DQNAgent()
    load_model(trained_agent, loaded_model)

    state = board_to_state(board)

    action = select_action(state, trained_agent, 45, 0)

    x1, y1, x2, y2 = decode_action(action, board)

    if is_valid_move(board, x1, y1, x2, y2, PLAYER1_SYMBOL):
        move_piece(board, x1, y1, x2, y2, "player1")
        capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_opponent")
        capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_own")
        capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_opponent")
        capture_after_move(board, PLAYER1_SYMBOL, PLAYER2_SYMBOL, "capture_own")

    return board, x1, y1, x2, y2


def player2_human_move(board, x1, y1, x2, y2):
    if is_valid_move(board, x1, y1, x2, y2, '●'):
        move_piece(board, x1, y1, x2, y2, "player2")
        capture_after_move(board, '●', '▲', "capture_opponent")
        capture_after_move(board, '●', '▲', "capture_own")
        capture_after_move(board, '●', '▲', "capture_opponent")
        capture_after_move(board, '●', '▲', "capture_own")
        return board, True
    else:
        print("Invalid move. Try again.")
        return board, False


def player1_human_move(board, x1, y1, x2, y2):
    if is_valid_move(board, x1, y1, x2, y2, '▲'):
        move_piece(board, x1, y1, x2, y2, "player1")
        capture_after_move(board, '▲', '●', "capture_opponent")
        capture_after_move(board, '▲', '●', "capture_own")
        return board, True
    else:
        print("Invalid move. Try again.")
        return board, False
