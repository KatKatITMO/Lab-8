import cv2
import numpy as np
'''Возьмите приложенное изображение из папки images/variant-<номер варианта>.jpg/png
и выполните его преобразование согласно вашему варианту.'''

image = cv2.imread("C:/PPROG LABS/test.jpeg")

if image is None:
    print("Ошибка загрузки изображения")
else:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    blue_extracted = cv2.bitwise_and(image, image, mask=mask)
    
    cv2.imshow('Blue Objects (HSV Masking)', blue_extracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''# Распечатайте изображение метки на листе бумаги и расместите его на поверхности. 
# Используя камеру, захватите поверхность с меткой и реализуйте алгоритм её отслеживания.'''
'''# Модифицируйте программу из п. 2, согласно вашему варианту.
# Сделайте проверку на попадание метки в область на экране - правая половина'''

'''# доп.задание - Возьмите приложенное изображение мухи (fly64.png) и наложите его на кадр программы 
из п. 2 таким образом, чтобы центр мухи совпадал с центром метки'''


import cv2
import numpy as np


overlay_img = cv2.imread('fly64.jpg', cv2.IMREAD_UNCHANGED) 
if overlay_img is None:
    print("Ошибка: не удалось загрузить изображение fly64.jpg")
    exit()

cap = cv2.VideoCapture(0)
tracking = False  
target_circle = None 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    mid_x = width // 2 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )

    display_frame = frame.copy()  
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        if not tracking:
            target_circle = circles[0, 0]
            tracking = True
        
        if tracking and target_circle is not None:
            min_dist = float('inf')
            closest_circle = None
            x_prev, y_prev, r_prev = target_circle
            
            for circle in circles[0, :]:
                x, y, r = circle
                dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_circle = circle
            
            if closest_circle is not None:
                target_circle = closest_circle
                x, y, r = target_circle
                
                img_size = r * 2
                resized_img = cv2.resize(overlay_img, (img_size, img_size))
                
                y1, y2 = y - r, y + r
                x1, x2 = x - r, x + r
                
                if y1 >= 0 and y2 <= height and x1 >= 0 and x2 <= width:
        
                    if overlay_img.shape[2] == 4:
                        alpha = resized_img[:, :, 3] / 255.0
                        for c in range(0, 3):
                            display_frame[y1:y2, x1:x2, c] = (
                                display_frame[y1:y2, x1:x2, c] * (1 - alpha) + 
                                resized_img[:, :, c] * alpha
                            )
                    else:
                        display_frame[y1:y2, x1:x2] = resized_img
                
                in_right_half = x > mid_x
                position_text = "Right half" if in_right_half else "Left half"
                cv2.putText(display_frame, position_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                tracking = False 
    else:
        tracking = False 

    cv2.line(display_frame, (mid_x, 0), (mid_x, height), (255, 255, 255), 1)
    
    cv2.imshow("Circle Tracking with Image", display_frame)
    
    if cv2.waitKey(1) == 32: 
        break

cap.release()
cv2.destroyAllWindows()


