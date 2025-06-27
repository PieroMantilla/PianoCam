import cv2
import mediapipe as mp
import pygame

# Inicializar pygame y cargar sonidos
pygame.init()
note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
sounds = {note: pygame.mixer.Sound(f'notes/{note}.wav') for note in note_names}

# Definir tamaño de teclas
WIDTH = 700
HEIGHT = 480
key_width = WIDTH // len(note_names)

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Iniciar cámara
cap = cv2.VideoCapture(0)

pressed_note = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Dibujar piano
    for i, note in enumerate(note_names):
        x = i * key_width
        color = (255, 255, 255)
        if pressed_note == note:
            color = (180, 180, 255)
        cv2.rectangle(frame, (x, HEIGHT - 120), (x + key_width, HEIGHT), color, -1)
        cv2.rectangle(frame, (x, HEIGHT - 120), (x + key_width, HEIGHT), (0, 0, 0), 2)
        cv2.putText(frame, note, (x + 10, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Detectar dedo
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            index_finger = hand.landmark[8]  # Dedo índice
            h, w, _ = frame.shape
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            if y > HEIGHT - 120:
                note_index = x // key_width
                if 0 <= note_index < len(note_names):
                    note = note_names[note_index]
                    if pressed_note != note:
                        sounds[note].play()
                        pressed_note = note
                else:
                    pressed_note = None
            else:
                pressed_note = None

    else:
        pressed_note = None

    cv2.imshow("Piano con la mano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()