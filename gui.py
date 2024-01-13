import os

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QGridLayout, QListWidget, QCheckBox, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QLabel
import sys

from main import ai_move, player2_human_move, player1_human_move
from play import play_game, player1_pieces, player2_pieces
from events import win_rate_updated_event
from utils import check_game_over

PLAYER1_SYMBOL = '▲'
PLAYER2_SYMBOL = '●'


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()


class MainWindow(QWidget):
    update_win_rates_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()

        self.current_round = 1
        self.selected_piece = None
        self.current_board = [[' ' for _ in range(7)] for _ in range(7)]
        self.current_player = PLAYER1_SYMBOL

        self.setWindowTitle("Strategic Board Game")
        self.setGeometry(100, 100, 1300, 600)

        main_layout = QHBoxLayout(self)

        self.game_status_label = QLabel("")  # game_status_label'ı tanımla
        main_layout.addWidget(self.game_status_label)

        self.grid_layout = QGridLayout()
        main_layout.addLayout(self.grid_layout, 60)

        train_layout = QVBoxLayout()
        main_layout.addLayout(train_layout, 20)

        train_header_label = QLabel("Train Agent")
        train_header_label.setStyleSheet("font-size: 18px; font-weight: bold; text-transform: uppercase;")
        train_header_label.setAlignment(Qt.AlignCenter)
        train_layout.addWidget(train_header_label)

        play_layout = QVBoxLayout()
        main_layout.addLayout(play_layout, 20)

        play_header_label = QLabel("Play Game")
        play_header_label.setStyleSheet("font-size: 18px; font-weight: bold; text-transform: uppercase;")
        play_header_label.setAlignment(Qt.AlignCenter)
        play_layout.addWidget(play_header_label)

        self.play_options_label = QLabel("Play Options:")
        play_layout.addWidget(self.play_options_label)

        self.play_options_combobox = QComboBox()
        self.play_options_combobox.addItems(
            ["Agent vs Human", "Human vs Human"])
        play_layout.addWidget(self.play_options_combobox)

        self.play_model_dropdown_label = QLabel("Load/Select Model:")
        play_layout.addWidget(self.play_model_dropdown_label)

        self.play_model_dropdown = QComboBox()
        play_layout.addWidget(self.play_model_dropdown)

        self.play_with_trained_agent_button = QPushButton("Start Game")
        self.play_with_trained_agent_button.clicked.connect(self.play_with_trained_agent)
        play_layout.addWidget(self.play_with_trained_agent_button)

        self.move_list = QListWidget()
        play_layout.addWidget(self.move_list)

        self.train_options_label = QLabel("Train Options:")
        train_layout.addWidget(self.train_options_label)

        self.train_options_combobox = QComboBox()
        self.train_options_combobox.addItems(
            ["Agent vs Random", "Agent vs Agent", "Random vs Random"])
        train_layout.addWidget(self.train_options_combobox)

        self.model_dropdown_label = QLabel("Load/Select Model:")
        train_layout.addWidget(self.model_dropdown_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("No Load")  # Add "Don't Load" option
        train_layout.addWidget(self.model_dropdown)
        self.populate_model_dropdown()

        self.save_model_as_label = QLabel("Save model as (empty -> no save)")
        train_layout.addWidget(self.save_model_as_label)

        self.save_model_as_textbox = QLineEdit()
        train_layout.addWidget(self.save_model_as_textbox)

        self.moves_per_round_label = QLabel("Moves per round:")
        train_layout.addWidget(self.moves_per_round_label)

        self.moves_per_round_combobox = QComboBox()
        self.moves_per_round_combobox.addItems(
            ["1", "2"])
        train_layout.addWidget(self.moves_per_round_combobox)

        self.max_rounds_per_game_label = QLabel("Max rounds per game:")
        train_layout.addWidget(self.max_rounds_per_game_label)

        self.max_rounds_per_game_combobox = QComboBox()
        self.max_rounds_per_game_combobox.addItems(
            ["50", "100", "200", "400"])
        train_layout.addWidget(self.max_rounds_per_game_combobox)

        self.games_label = QLabel("Games: ")
        train_layout.addWidget(self.games_label)

        self.games_combobox = QComboBox()
        self.games_combobox.addItems(
            ["100", "300", "500", "1000", "2000", "3000", "5000", "10000"])
        train_layout.addWidget(self.games_combobox)

        self.train_agent_button = QPushButton("Train Agent")
        self.train_agent_button.clicked.connect(self.train_agent)
        train_layout.addWidget(self.train_agent_button)

        self.test_trained_agent_button = QPushButton("Test Trained Agent vs Random")
        self.test_trained_agent_button.clicked.connect(self.test_trained_agent)
        train_layout.addWidget(self.test_trained_agent_button)

        self.training_status_label = QLabel("")
        main_layout.addWidget(self.training_status_label)

        self.player1_win_rate_label = QLabel("Player 1 Win Rate: ")
        train_layout.addWidget(self.player1_win_rate_label)

        self.player2_win_rate_label = QLabel("Player 2 Win Rate: ")
        train_layout.addWidget(self.player2_win_rate_label)

        self.draw_rate_label = QLabel("Draw Rate:")
        train_layout.addWidget(self.draw_rate_label)

        win_rate_updated_event.connect(self.update_win_rates)

        self.setup_board()

    def start_game(self):
        self.move_list.clear()
        self.current_round = 1
        self.current_player = PLAYER1_SYMBOL
        if self.play_options_combobox.currentText() == "Agent vs Human":
            self.make_ai_move()

    def make_ai_move(self):
        loaded_model = self.play_model_dropdown.currentText()
        if self.current_player == PLAYER1_SYMBOL:
            self.current_board, x1, y1, x2, y2 = ai_move(self.current_board, loaded_model)
            self.update_board(self.current_board)
            move_description = f"Round {self.current_round}: Player 1 (AI) move from {x1},{y1} to {x2},{y2}"
            self.update_move_list(move_description)
            if not self.check_game_end():
                self.current_player = PLAYER2_SYMBOL
                self.update_game_status("Player 2's Turn")

    def update_game_status(self, status):
        self.game_status_label.setText(status)

    def setup_board(self):
        self.grid_labels = [[ClickableLabel(self) for _ in range(7)] for _ in range(7)]
        for row in range(7):
            for col in range(7):
                label = self.grid_labels[row][col]
                label.setFixedSize(80, 80)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("QLabel { background-color: white; border: 1px solid black; }")
                label.clicked.connect(lambda _=None, row=row, col=col: self.label_clicked(row, col))
                self.grid_layout.addWidget(label, row, col)

        for (x, y) in player1_pieces:
            self.current_board[x][y] = PLAYER1_SYMBOL
            self.update_label_pixmap(x, y, PLAYER1_SYMBOL)

        for (x, y) in player2_pieces:
            self.current_board[x][y] = PLAYER2_SYMBOL
            self.update_label_pixmap(x, y, PLAYER2_SYMBOL)

    def update_label_pixmap(self,    row, col, player_symbol):
        label = self.grid_labels[row][col]
        if player_symbol == PLAYER1_SYMBOL:
            pixmap = QPixmap('assets/triangle-red-2.png')
        elif player_symbol == PLAYER2_SYMBOL:
            pixmap = QPixmap('assets/circle-blue.png')
        else:
            pixmap = QPixmap()
        label.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
        label.repaint()

    def label_clicked(self, row, col):
        if self.play_options_combobox.currentText() == "Human vs Human":
            self.make_human_move(row, col)
        else:
            if self.current_player == PLAYER2_SYMBOL:
                if self.selected_piece:
                    x1, y1 = self.selected_piece
                    print(x1, y1, row, col)
                    self.current_board, valid_move = player2_human_move(self.current_board, x1, y1, row, col)

                    if valid_move:
                        self.update_board(self.current_board)
                        move_description = f"Round {self.current_round}: Player 2 (HUMAN) move from {x1},{y1} to {row},{col}"
                        self.update_move_list(move_description)
                        self.end_round()

                        if not self.check_game_end():
                            self.update_game_status("Player 1's Turn")  # Oyun durumunu güncelle
                            self.current_player = PLAYER1_SYMBOL
                            self.make_ai_move()
                            self.selected_piece = None
                    else:
                        move_description = "Invalid move. Player 2, try again."
                        self.update_move_list(move_description)

                elif self.is_player2_piece(row, col):
                    self.selected_piece = (row, col)

    def is_player2_piece(self, row, col):
        return self.current_board[row][col] == PLAYER2_SYMBOL

    def is_player1_piece(self, row, col):
        return self.current_board[row][col] == PLAYER1_SYMBOL

    def update_board(self, board):
        print("Board is being updated with:", board)
        for row in range(7):
            for col in range(7):
                label = self.grid_labels[row][col]
                piece = board[row][col]

                if piece == PLAYER1_SYMBOL:
                    pixmap = QPixmap('assets/triangle-red-2.png')
                    label.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
                elif piece == PLAYER2_SYMBOL:
                    pixmap = QPixmap('assets/circle-blue.png')
                    label.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))
                else:
                    label.setPixmap(QPixmap())
                    label.setText("")

                label.setStyleSheet("QLabel { background-color: white; border: 1px solid black; }")
                label.repaint()

    def train_agent(self):
        number_of_move_per_round = int(self.moves_per_round_combobox.currentText())
        max_rounds_per_game = int(self.max_rounds_per_game_combobox.currentText())
        episodes = int(self.games_combobox.currentText())

        loaded_model = self.model_dropdown.currentText()

        save_model_as = self.save_model_as_textbox.text()  # Kaydedilecek model adını alın

        train_option = self.train_options_combobox.currentText()

        play_game(number_of_move_per_round, max_rounds_per_game, episodes, loaded_model, save_model_as, False,
                  train_option)

        self.update_load_list()

    def test_trained_agent(self):
        number_of_move_per_round = int(self.moves_per_round_combobox.currentText())
        max_rounds_per_game = int(self.max_rounds_per_game_combobox.currentText())
        episodes = int(self.games_combobox.currentText())

        loaded_model = self.model_dropdown.currentText()

        play_game(number_of_move_per_round, max_rounds_per_game, episodes, loaded_model, "", True, "Agent vs Random")

        self.update_load_list()

    def update_win_rates(self, player1_win_rate, player2_win_rate, draw_rate):
        self.player1_win_rate_label.setText(f"Player 1 Win Rate: {player1_win_rate}%")
        self.player2_win_rate_label.setText(f"Player 2 Win Rate: {player2_win_rate}%")
        self.draw_rate_label.setText(f"Draw Rate: {draw_rate}%")

    def play_with_trained_agent(self):
        self.reset_game()
        self.start_game()

    def update_move_list(self, move):
        self.move_list.addItem(move)
        self.move_list.scrollToBottom()

    def check_game_end(self):
        game_status = check_game_over(self.current_board, self.current_round, 200)
        if game_status != 9:
            if game_status == 0:
                self.update_game_status("It's a draw!")
            elif game_status == 1:
                self.update_game_status("Player 1 wins!")
            elif game_status == 2:
                self.update_game_status("Player 2 wins!")
            return True
        return False

    def update_load_list(self):
        self.model_dropdown.clear()
        self.model_dropdown.addItem("No Load")
        self.play_model_dropdown.clear()
        self.populate_model_dropdown()

    def end_round(self):
        self.current_round += 1
        print(f"Round {self.current_round} completed.")

    def reset_game(self):
        self.current_round = 1
        self.current_board = [[' ' for _ in range(7)] for _ in range(7)]
        self.selected_piece = None
        self.current_player = PLAYER1_SYMBOL
        self.move_list.clear()

        self.setup_board()
        self.update_game_status("Game reset. Player 1's Turn")

    def populate_model_dropdown(self):
        model_folder = "./models"  # Change this to the path of your model folder
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]

        if model_files:
            self.model_dropdown.addItems(model_files)
            self.play_model_dropdown.addItems(model_files)
        else:
            self.model_dropdown.addItem("No .pth files found")
            self.play_model_dropdown.addItem("No .pth files found")

    def make_human_move(self, row, col):
        if self.current_player == PLAYER1_SYMBOL:
            if self.is_player1_piece(row, col):
                self.selected_piece = (row, col)
            elif self.selected_piece:
                x1, y1 = self.selected_piece
                self.current_board, valid_move = player1_human_move(self.current_board, x1, y1, row, col)

                if valid_move:
                    self.update_board(self.current_board)
                    move_description = f"Round {self.current_round}: Player 1 (HUMAN) move from {x1},{y1} to {row},{col}"
                    self.update_move_list(move_description)
                    self.end_round()

                    if not self.check_game_end():
                        self.current_player = PLAYER2_SYMBOL
                        self.update_game_status("Player 2's Turn")
                        self.selected_piece = None
                else:
                    move_description = "Invalid move. Player 1, try again."
                    self.update_move_list(move_description)
        elif self.current_player == PLAYER2_SYMBOL:
            if self.is_player2_piece(row, col):
                self.selected_piece = (row, col)
            elif self.selected_piece:
                x1, y1 = self.selected_piece
                self.current_board, valid_move = player2_human_move(self.current_board, x1, y1, row, col)

                if valid_move:
                    self.update_board(self.current_board)
                    move_description = f"Round {self.current_round}: Player 2 (HUMAN) move from {x1},{y1} to {row},{col}"
                    self.update_move_list(move_description)
                    self.end_round()

                    if not self.check_game_end():
                        self.current_player = PLAYER1_SYMBOL
                        self.update_game_status("Player 1's Turn")
                        self.selected_piece = None
                else:
                    move_description = "Invalid move. Player 2, try again."
                    self.update_move_list(move_description)

        self.update_game_status(f"{self.current_player}'s Turn")


def run_app():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


run_app()
