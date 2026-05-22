import cv2
import serial
import time
import numpy as np

PUERTO_SERIAL = 'COM5' 

try:
    arduino = serial.Serial(PUERTO_SERIAL, 9600)
    time.sleep(2)
    print("Conexion serial establecida con el Arduino de manera exitosa.")
except Exception as e:
    print(f"Alerta: No se detecto un Arduino en {PUERTO_SERIAL}. Modo simulacion activo.")
    arduino = None

cap = cv2.VideoCapture(0)
print("Iniciando camara por clasificacion de color...")

# Variable para registrar el último gesto enviado y no repetir ráfagas
ultimo_gesto = "0"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None: 
        continue
    
    frame = cv2.flip(frame, 1)
    
    alto, ancho = frame.shape[:2]
    x1, y1 = int(ancho * 0.3), int(alto * 0.25)
    x2, y2 = int(ancho * 0.7), int(alto * 0.75)
    
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    bajo_piel = np.array([0, 30, 60], dtype=np.uint8)
    alto_piel = np.array([20, 150, 255], dtype=np.uint8)
    
    mascara = cv2.inRange(hsv, bajo_piel, alto_piel)
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=2)
    
    pixeles_piel = cv2.countNonZero(mascara)
    
    gesto = "0"
    
    if pixeles_piel > 18000:
        gesto = "1" # PUERTA 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(frame, "PUERTA 1: ABIERTA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
    elif 4000 < pixeles_piel <= 18000:
        gesto = "3" # PUERTA 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, "PUERTA 2: ABIERTA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
    else:
        gesto = "0" # ESPERA
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, "SISTEMA EN ESPERA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # REGLA DE TRANSMISIÓN ÚNICA: Solo envía si el estado cambió
    if gesto != ultimo_gesto:
        if arduino and arduino.is_open:
            try:
                arduino.write(f"{gesto}\n".encode())
                ultimo_gesto = gesto # Actualizamos el estado
            except Exception:
                pass
        
    cv2.imshow("PIA Mecatronica - Control por Vision", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
if arduino: 
    arduino.close()
cv2.destroyAllWindows()