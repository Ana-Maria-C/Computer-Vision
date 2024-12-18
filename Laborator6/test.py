import os


def count_objects_yolo(labels_folder, target_class_id=0):
    """
    Numără câte obiecte dintr-o anumită clasă există în toate fișierele de etichete YOLO.

    :param labels_folder: Calea către folderul cu fișierele de etichete (.txt).
    :param target_class_id: ID-ul clasei pe care vrem să o numărăm.
    :return: Numărul total de obiecte găsite.
    """
    total_count = 0

    # Parcurgem toate fișierele din folder
    for file_name in os.listdir(labels_folder):
        if file_name.endswith(".txt"):  # Asigură-te că sunt fișiere de etichete YOLO
            file_path = os.path.join(labels_folder, file_name)

            with open(file_path, "r") as file:
                for line in file:
                    class_id = int(line.split()[0])  # Extragem primul element (class_id)
                    if class_id == target_class_id:
                        total_count += 1

    return total_count


# Setează calea către folderul cu etichete YOLO
labels_folder = r"C:\Users\Ana-Maria\OneDrive\Desktop\Laborator6\YOLO"
target_class_id = 0  # ID-ul clasei pe care vrei să o numări

# Apelează funcția
total_objects = count_objects_yolo(labels_folder, target_class_id)
print(f"Numărul total de obiecte din clasa {target_class_id}: {total_objects}")
