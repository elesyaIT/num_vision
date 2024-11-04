import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import pytesseract

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="4CWzb9M5NKv0kw9lMCGO"
)

image_path = "/home/ivan/number_vision/car/10.jpg"
result = CLIENT.infer(image_path, model_id="masking-license-plates/1")
print("result", result)

image = cv2.imread(image_path)
img_height, img_width = image.shape[:2]
print(f"Размер изображения: {img_width}x{img_height}")

if result['predictions']:
    for prediction in result['predictions']:
        x, y = prediction['x'], prediction['y']
        width, height = prediction['width'], prediction['height']
        confidence = prediction['confidence']
        print(confidence)

        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y - height / 2)
        x3, y3 = int(x + width / 2), int(y + height / 2)
        x4, y4 = int(x - width / 2), int(y + height / 2)

        # Определяем массив углов для коррекции перспективы
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")

        # Определяем ширину и высоту прямоугольника после трансформации
        plate_width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        plate_height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))

        # Указываем целевые точки для выпрямления
        dst = np.array([
            [0, 0],
            [plate_width - 1, 0],
            [plate_width - 1, plate_height - 1],
            [0, plate_height - 1]], dtype="float32")

        # Преобразование перспективы
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (plate_width, plate_height))

        # Сохраняем выровненное изображение номерного знака
        output_path = f"/home/ivan/my_vis/car/aligned_plate_{555}.jpeg"
        cv2.imwrite(output_path, warped)
        print(f"Обрезанный и выровненный номерной знак сохранён как {output_path}")

        # Путь к выровненному изображению для OCR
        image_path3 = output_path

        # Конвертация выровненного изображения в оттенки серого для OCR
        image = cv2.imread(image_path3, cv2.IMREAD_GRAYSCALE)

        # Увеличение контрастности с помощью бинаризации
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Настройка параметров OCR
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Распознавание текста
        recognized_text = pytesseract.image_to_string(binary_image, config=custom_config)
        print(f"Распознанный текст: {recognized_text.strip()}")

else:
    print("Номерной знак не найден.")
