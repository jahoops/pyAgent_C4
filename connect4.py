# connect4.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 6 rows, 7 columns
        self.current_player = 1  # Player 1 starts

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
        return self

    def make_move(self, action):
        """
        Makes a move in the specified column (action).
        Returns True if the move is valid, False otherwise.
        """
        if action < 0 or action >= 7:
            return False  # Invalid column

        for row in reversed(range(6)):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                self.current_player = 3 - self.current_player  # Switch player
                return True
        return False  # Column is full

    def get_legal_moves(self):
        """
        Returns a list of valid columns where a move can be made.
        """
        return [c for c in range(7) if self.board[0][c] == 0]

    def is_full(self):
        """
        Checks if the board is full.
        """
        return np.all(self.board != 0)

    def check_winner(self):
        """
        Checks for a winner.
        Returns the player number if there's a winner, 0 otherwise.
        """
        # Check horizontal locations for win
        for row in range(6):
            for col in range(7 - 3):
                if (self.board[row][col] == self.board[row][col + 1] ==
                    self.board[row][col + 2] == self.board[row][col + 3] != 0):
                    return self.board[row][col]

        # Check vertical locations for win
        for col in range(7):
            for row in range(6 - 3):
                if (self.board[row][col] == self.board[row + 1][col] ==
                    self.board[row + 2][col] == self.board[row + 3][col] != 0):
                    return self.board[row][col]

        # Check positively sloped diagonals
        for row in range(6 - 3):
            for col in range(7 - 3):
                if (self.board[row][col] == self.board[row + 1][col + 1] ==
                    self.board[row + 2][col + 2] == self.board[row + 3][col + 3] != 0):
                    return self.board[row][col]

        # Check negatively sloped diagonals
        for row in range(3, 6):
            for col in range(7 - 3):
                if (self.board[row][col] == self.board[row - 1][col + 1] ==
                    self.board[row - 2][col + 2] == self.board[row - 3][col + 3] != 0):
                    return self.board[row][col]

        return 0  # No winner

    def is_terminal(self):
        """
        Determines if the game has ended.
        Returns True if there's a winner or the board is full, False otherwise.
        """
        return self.check_winner() != 0 or self.is_full()

    def game_result(self):
        """
        Returns the result of the game from the perspective of the last player to move.
        1 if Player 1 wins,
        -1 if Player 2 wins,
        0 for a draw.
        """
        winner = self.check_winner()
        if winner == 1:
            return 1
        elif winner == 2:
            return -1
        else:
            return 0

    def clone(self):
        """
        Creates a deep copy of the game environment.
        """
        new_env = Connect4()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Connect4Net(nn.Module):
    def __init__(self, action_dim):
        super(Connect4Net, self).__init__()
        channels = 64
        self.initial_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(8)])

        self.fc_common = nn.Linear(channels * 6 * 7, 256)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_common(x))
        log_policy = F.log_softmax(self.policy_head(x), dim=1)  # Policy head
        value = torch.tanh(self.value_head(x))  # Value head
        return log_policy, value
    def __init__(self, action_dim):
        super(Connect4Net, self).__init__()
        channels = 64
        self.initial_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(8)])

        self.fc_common = nn.Linear(channels * 6 * 7, 256)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.initial_conv(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_common(x))
        log_policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return log_policy, value
    def game_result(self):
        """
        Returns the result of the game from the perspective of the last player to move.
        1 if Player 1 wins,
        -1 if Player 2 wins,
        0 for a draw.
        """
        winner = self.check_winner()
        if winner == 1:
            return 1
        elif winner == 2:
            return -1
        else:
            return 0