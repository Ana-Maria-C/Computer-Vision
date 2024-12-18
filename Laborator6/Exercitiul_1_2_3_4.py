from ultralytics import YOLO
import cv2


def count_persons_in_image(image_path):
    model = YOLO('yolov8n.pt')
    img = cv2.imread(image_path)
    results = model(img)
    num_persons = sum(1 for box in results[0].boxes.data if int(box[-1]) == 0)
    print(f'Număr persoane detectate: {num_persons}')


def detect_objects_in_image(image_path):
    model = YOLO('yolov8n.pt')
    img = cv2.imread(image_path)
    results = model(img)
    annotated_frame = results[0].plot()
    cv2.imshow('Detecție Obiecte', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_top_objects_in_video(video_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        # Sortează obiectele după scoruri și ia top 3-4
        top_objects = sorted(results[0].boxes.data, key=lambda x: x[4], reverse=True)[:4]
        annotated_frame = results[0].plot(boxes=top_objects)

        cv2.imshow('Top Detecții Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_objects_in_image("Exercitiul3.jpg")
    count_persons_in_image("Exercitiul3.jpg")
    detect_top_objects_in_video("sample.mp4")

