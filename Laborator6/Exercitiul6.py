import os
import random
import shutil
import cv2
from ultralytics import YOLO
import glob
import sys


def split_dataset(images_folder, boxes_folder, output_folder, train_ratio=0.7):
    """
    Împarte dataset-ul în 70% train și 30% test.
    """
    # Structura folderelor
    train_images = os.path.join(output_folder, "train", "images")
    train_labels = os.path.join(output_folder, "train", "labels")
    test_images = os.path.join(output_folder, "test", "images")
    test_labels = os.path.join(output_folder, "test", "labels")

    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)

    # Listează toate imaginile
    images = sorted(os.listdir(images_folder))
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    # Copiază și convertește datele
    for img_file in train_files:
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(train_images, img_file))
        convert_boxes_to_yolo(img_file, boxes_folder, train_labels)

    for img_file in test_files:
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(test_images, img_file))
        convert_boxes_to_yolo(img_file, boxes_folder, test_labels)


def convert_boxes_to_yolo(img_file, boxes_folder, dest_labels_folder):
    """
      Înlocuiește generarea adnotărilor YOLO cu citirea unui fișier text existent.
      """
    # Calea către fișierul de imagine
    box_path = os.path.join(boxes_folder, img_file)

    # Calea către fișierul de etichete YOLO de destinație
    label_file = os.path.join(dest_labels_folder, img_file.replace(".png", ".txt"))

    # Calea către fișierul text cu adnotări (schimbăm extensia la .txt)
    annotation_file = os.path.join("YOLO", img_file.replace(".PNG", ".txt"))

    # Verificăm dacă fișierul de imagine există
    if not os.path.exists(box_path):
        print(f"⚠️ Fișier lipsă: {box_path}")
        return

    # Verificăm dacă fișierul de adnotări există în folderul specificat
    if not os.path.exists(annotation_file):
        print(f"⚠️ Fișier de adnotări lipsă: {annotation_file}")
        return

    # Citim conținutul fișierului de adnotări
    with open(annotation_file, "r") as f:
        annotations = f.read()

    # Scriem conținutul citit din fișierul de adnotări în fișierul de etichete YOLO de destinație
    with open(label_file, "w") as f:
        f.write(annotations)

    print(f"✅ Fișierul de etichete a fost creat: {label_file}")


def train_yolo_model(dataset_path, epochs=10):
    """
    Antrenează modelul YOLO pe setul de date.
    """
    print("Antrenez modelul YOLO...")
    model = YOLO("yolov8n.pt")
    model.train(data=os.path.join(dataset_path, "dataset.yaml"), epochs=epochs, imgsz=640)
    print("Antrenarea s-a finalizat!")


def evaluate_model(model_path, test_images_folder):

    """
    Evaluează modelul pe setul de test și numără mașinile detectate.
    """
    print("Evaluez modelul pe setul de test...")
    model = YOLO(model_path)
    images = sorted(os.listdir(test_images_folder))

    total_detected = 0
    for img_file in images:
        img_path = os.path.join(test_images_folder, img_file)
        results = model(img_path)
        cars = sum(1 for box in results[0].boxes.data if int(box[-1]) == 0)
        total_detected += cars
        print(f"Imagine: {img_file} -> Mașini detectate: {cars}")

    print(f"Total mașini detectate pe setul de test: {total_detected}")


def create_yaml(dataset_path):
    """
    Creează fișierul dataset.yaml pentru YOLO.
    """
    # Obține calea absoluta pentru train și test
    train_images = os.path.abspath(os.path.join(dataset_path, "train", "images"))
    test_images = os.path.abspath(os.path.join(dataset_path, "test", "images"))

    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_images}\n")
        f.write(f"val: {test_images}\n")
        f.write("nc: 1\n")
        f.write("names: ['car']\n")
    print(f"Fișierul dataset.yaml a fost creat: {yaml_path}")
    return yaml_path


def find_best_model():
    """
    Găsește fișierul 'best.pt' în directoarele YOLO generate automat.
    """
    model_paths = glob.glob("runs/detect/train*/weights/best.pt")
    if model_paths:
        print(f"✅ Model găsit: {model_paths[0]}")
        return model_paths[0]
    else:
        raise FileNotFoundError("❌ Nu am găsit niciun fișier best.pt.")

def main():
    log_file = "output_log.txt"
    sys.stdout = open(log_file, "w")

    try:
        # Căile dataset-ului
        images_folder = r"C:\Users\Ana-Maria\Downloads\archive\images"
        boxes_folder = r"C:\Users\Ana-Maria\Downloads\archive\boxes"
        dataset_path = "dataset_yolo"

        # Pas 1: Split dataset
        print("Împart dataset-ul în 70% train și 30% test...")
        split_dataset(images_folder, boxes_folder, dataset_path)
        print("Dataset împărțit cu succes!")

        # Pas 3: Antrenează modelul YOLO
        train_yolo_model(dataset_path, epochs=10)

        # Pas 4: Găsește fișierul 'best.pt'
        model_path = find_best_model()

        # Pas 5: Evaluează modelul pe setul de test
        test_images_folder = os.path.join(dataset_path, "test", "images")
        evaluate_model(model_path, test_images_folder)

    finally:
        # Reseteaza stdout
        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()