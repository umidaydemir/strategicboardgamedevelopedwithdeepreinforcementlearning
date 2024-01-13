class WinRateUpdatedEvent:
    def __init__(self):
        self.listeners = []

    def connect(self, listener):
        self.listeners.append(listener)

    def emit(self, player1_win_rate, player2_win_rate, draw_rate):
        for listener in self.listeners:
            listener(player1_win_rate, player2_win_rate, draw_rate)


win_rate_updated_event = WinRateUpdatedEvent()
