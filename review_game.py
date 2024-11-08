# review_game.py
import pygame
import pickle
import time

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
AGENT1_COLOR = (255, 0, 0)
AGENT2_COLOR = (0, 0, 255)

def draw_board(screen, board):
    screen.fill(BACKGROUND_COLOR)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            color = EMPTY_COLOR
            if board[row][col] == 1:
                color = AGENT1_COLOR
            elif board[row][col] == 2:
                color = AGENT2_COLOR
            pygame.draw.circle(screen, color, (
                col * (CIRCLE_RADIUS * 2 + CIRCLE_PADDING) + CIRCLE_RADIUS + CIRCLE_PADDING,
                row * (CIRCLE_RADIUS * 2 + CIRCLE_PADDING) + CIRCLE_RADIUS + CIRCLE_PADDING
            ), CIRCLE_RADIUS)
    pygame.display.flip()

def review_games():
    with open("random_games.pkl", "rb") as f:
        games = pickle.load(f)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Connect 4 Game Review")
    clock = pygame.time.Clock()

    for game in games:
        for move in game:
            agent, action, board = move
            draw_board(screen, board)
            time.sleep(1)  # Wait for 1 second between moves
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        time.sleep(DELAY_BETWEEN_GAMES)  # Wait for 5 seconds between games

    pygame.quit()

if __name__ == "__main__":
    review_games()