import cv2
import pytesseract

# Path to Tesseract executable (update this if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as necessary

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_image(image):
    # Resize image to increase the text clarity for Tesseract
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply a slight Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Increase the contrast of the image for clearer text
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    
    return image

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame for enhanced OCR results
    processed_frame = preprocess_image(frame)

    # OCR with Tesseract on the processed frame
    custom_config = r'--oem 3 --psm 6'  # LSTM OCR Engine, assumes a single block of text
    text = pytesseract.image_to_string(processed_frame, config=custom_config)

    # Display the recognized text on the frame in real-time
    cv2.putText(frame, "Detected Text: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Show the original frame with text overlay
    cv2.imshow("Real-Time Text Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
