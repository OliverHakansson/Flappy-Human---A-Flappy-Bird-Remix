import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

def get_y_from_x(x, b, m):
    return int(round(m * x + b))

# Initialize variables
wasAboveShoulders = False
aboveShoulders = False
numFlaps = 0
prevPosition = 0
armVelocity = 0
lastTime = time.time()

# Read an initial frame to get dimensions
ret, frame = cap.read()
if ret:
    height, width, _ = frame.shape
else:
    raise RuntimeError("Failed to read from webcam.")

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

gravity = 0.5
flap_strength = -10
bird_y = HEIGHT // 2
bird_velocity = 0
score = 0

# Load bird image
bird = pygame.image.load("bird.png")
bird = pygame.transform.scale(bird, (50, 35))

# Obstacles
obstacle_width = 70
obstacle_gap = 150
obstacles = []

def create_obstacle():
    top_height = random.randint(50, HEIGHT - obstacle_gap - 50)
    bottom_height = HEIGHT - top_height - obstacle_gap
    return [WIDTH, top_height, bottom_height]

def move_obstacles():
    global obstacles, score
    for obstacle in obstacles:
        obstacle[0] -= 5
    obstacles = [obs for obs in obstacles if obs[0] > -obstacle_width]
    if len(obstacles) == 0 or obstacles[-1][0] < WIDTH - 350:
        obstacles.append(create_obstacle())
    for obs in obstacles:
        if obs[0] == WIDTH // 4:
            score += 1

# Game states
START, PLAYING, GAME_OVER = 0, 1, 2
game_state = START

# Fonts
font = pygame.font.Font(None, 36)

def draw_text(text, x, y):
    render = font.render(text, True, (255, 255, 255))
    screen.blit(render, (x, y))

def setFlapState(height, width, b, m, results):
    global aboveShoulders, flap_strength, lastTime, prevPosition  

    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    left_wrist_above = (left_wrist.y * height) < get_y_from_x(left_wrist.x * width, b, m)
    right_wrist_above = (right_wrist.y * height) < get_y_from_x(right_wrist.x * width, b, m)

    aboveShoulders = left_wrist_above and right_wrist_above

    # Compute wrist average position
    wrist_avg_y = (left_wrist.y + right_wrist.y) / 2

    # Compute arm velocity
    currentTime = time.time()
    timeDiff = currentTime - lastTime
    if timeDiff > 0:
        armVelocity = abs(wrist_avg_y - prevPosition) / timeDiff
    else:
        armVelocity = 0

    # Set flap strength based on arm speed
    flap_strength = armVelocity*-3

    # Update time and position
    lastTime = currentTime  
    prevPosition = wrist_avg_y  

running = True
while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        height, width, _ = frame.shape
        
        try:
            left_shoulder_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder_lm = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = (int(left_shoulder_lm.x * width), int(left_shoulder_lm.y * height))
            right_shoulder = (int(right_shoulder_lm.x * width), int(right_shoulder_lm.y * height))
            
            m = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0])
            b = left_shoulder[1] - m * left_shoulder[0]
            
            wasAboveShoulders = aboveShoulders
            setFlapState(height, width, b, m, results)
            
            if game_state == PLAYING and wasAboveShoulders and not aboveShoulders:
                numFlaps += 1
                bird_velocity = flap_strength
                flap_strength = 0
                print("Flap Num:", numFlaps)
            
            cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 2)
        except IndexError:
            pass  
    
    cv2.imshow("Flappy Human", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
    
    screen.fill((135, 206, 235))  # Sky blue background
    
    if game_state == START:
        draw_text("Press SPACE to Start", WIDTH // 4, HEIGHT // 2)
    elif game_state == PLAYING:

        bird_y += bird_velocity
        bird_velocity += gravity
        bird_y = max(0, min(HEIGHT - bird.get_height(), bird_y))
        screen.blit(bird, (WIDTH // 4, bird_y))
        move_obstacles()
        draw_text(f"Score: {score}", 10, 10)
        for obs in obstacles:
            pygame.draw.rect(screen, (0, 255, 0), (obs[0], 0, obstacle_width, obs[1]))
            pygame.draw.rect(screen, (0, 255, 0), (obs[0], HEIGHT - obs[2], obstacle_width, obs[2]))
            if obs[0] < WIDTH // 4 + 50 and obs[0] + obstacle_width > WIDTH // 4 and (bird_y < obs[1] or bird_y + 35 > HEIGHT - obs[2]):
                game_state = GAME_OVER
    elif game_state == GAME_OVER:
        draw_text("Game Over! Press SPACE to Restart", WIDTH // 6, HEIGHT // 2)
        draw_text(f"Final Score: {score}", WIDTH // 3, HEIGHT // 3)
    
    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if game_state == START:
                game_state = PLAYING
                score = 0
            elif game_state == GAME_OVER:
                game_state = START
                bird_y = HEIGHT // 2
                bird_velocity = 0
                obstacles = []
                score = 0

cap.release()
cv2.destroyAllWindows()
pygame.quit()
