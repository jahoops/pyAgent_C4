# review_game.py
import pygame
import pickle
import time
import random

# Constants
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 600
BOARD_ROWS = 6
BOARD_COLS = 7
CIRCLE_RADIUS = 40
CIRCLE_PADDING = 10
FPS = 1  # Frames per second (1 move per second)
DELAY_BETWEEN_GAMES = 5  # Seconds between games

# Colors
BACKGROUND_COLOR = (0, 0, 0)
EMPTY_COLOR = (255, 255, 255)

# Agent colors (consistent with markers assigned in evaluate_agents_q_vs_dqn.py)
AGENT_COLORS = {
    "Q-Learning Agent": (255, 0, 0),      # Red
    "DQN Agent": (0, 255, 0),             # Green
    "AlphaZero Agent": (0, 0, 255),       # Blue
}

def draw_board(screen, board):
    screen.fill(BACKGROUND_COLOR)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            marker = board[row][col]
            if marker == 0:
                color = EMPTY_COLOR
            else:
                # Map marker to color
                color = None
                for agent_name, agent_marker in AGENT_MARKERS.items():
                    if marker == agent_marker:
                        color = AGENT_COLORS[agent_name]
                        break
                if color is None:
                    color = EMPTY_COLOR  # Default to empty if marker not recognized

            pygame.draw.circle(screen, color, (
                col * (CIRCLE_RADIUS * 2 + CIRCLE_PADDING) + CIRCLE_RADIUS + CIRCLE_PADDING,
                row * (CIRCLE_RADIUS * 2 + CIRCLE_PADDING) + CIRCLE_RADIUS + CIRCLE_PADDING
            ), CIRCLE_RADIUS)
    pygame.display.flip()

def review_games():
    with open("alphazero_vs_q_games.pkl", "rb") as f:
        all_games = pickle.load(f)

    # Create a reverse mapping from marker to agent name
    global AGENT_MARKERS
    AGENT_MARKERS = {
        "Q-Learning Agent": 1,
        "DQN Agent": 2,
        "AlphaZero Agent": 3
    }

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Connect 4 Game Review")
    clock = pygame.time.Clock()

    # Randomly shuffle the games
    random.shuffle(all_games)

    for game in all_games:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        for move in game:
            agent_name, action, board = move
            draw_board(screen, board)
            time.sleep(1)  # Wait for 1 second between moves

        time.sleep(DELAY_BETWEEN_GAMES)  # Wait between games

    pygame.quit()

if __name__ == "__main__":
    review_games()