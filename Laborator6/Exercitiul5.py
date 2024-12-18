import xml.etree.ElementTree as ET
import cv2
import os


def parse_annotations(xml_path):
    """
    Parsează fișierul XML pentru a extrage datele despre vehiculele detectate.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_data = {}
    for track in root.findall('track'):
        track_id = track.attrib['id']
        label = track.attrib['label']

        for box in track.findall('box'):
            frame = int(box.attrib['frame'])
            if frame not in frame_data:
                frame_data[frame] = []

            frame_data[frame].append({
                "track_id": track_id,
                "label": label,
                "coordinates": {
                    "xtl": float(box.attrib['xtl']),
                    "ytl": float(box.attrib['ytl']),
                    "xbr": float(box.attrib['xbr']),
                    "ybr": float(box.attrib['ybr'])
                },
                "outside": int(box.attrib['outside']),  # 1 = vehiculul iese din cadru
            })

    return frame_data


def count_vehicles(frame_data):
    """
    Numără vehiculele (mașini și microbuze) pe fiecare cadru.
    """
    frame_counts = {}
    for frame, vehicles in frame_data.items():
        cars = sum(1 for v in vehicles if v["label"] == "car")
        minivans = sum(1 for v in vehicles if v["label"] == "minivan")
        frame_counts[frame] = {"cars": cars, "minivans": minivans}

    return frame_counts


def track_selected_vehicles(frame_data, images_folder, output_folder, selected_vehicle_ids):
    """
    Urmărește vehiculele selectate și salvează imaginile la intrarea și ieșirea acestora,
    marcând vehiculele urmărite cu chenare în jurul lor.
    """
    os.makedirs(output_folder, exist_ok=True)
    vehicle_tracks = {}

    for frame, vehicles in frame_data.items():
        image_path = os.path.join(images_folder, f"frame_{frame:06d}.png")
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)  # Încarcă imaginea pentru adnotări

        for vehicle in vehicles:
            track_id = vehicle["track_id"]
            if track_id not in selected_vehicle_ids:
                continue

            if track_id not in vehicle_tracks:
                vehicle_tracks[track_id] = {"enter_frame": frame, "exit_frame": None}

                # Marchează vehiculul la intrare
                coordinates = vehicle["coordinates"]
                xtl, ytl, xbr, ybr = int(coordinates["xtl"]), int(coordinates["ytl"]), int(coordinates["xbr"]), int(coordinates["ybr"])
                color = (0, 255, 0)  # Verde pentru vehiculele urmărite
                cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color, 2)
                cv2.putText(img, f"ID: {track_id}", (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Salvează imaginea de la intrare
                output_enter = os.path.join(output_folder, f"vehicle_{track_id}_enter_frame_{frame}.png")
                cv2.imwrite(output_enter, img)

            if vehicle["outside"] == 1:  # Vehiculul iese din cadru
                vehicle_tracks[track_id]["exit_frame"] = frame

                # Marchează vehiculul la ieșire
                coordinates = vehicle["coordinates"]
                xtl, ytl, xbr, ybr = int(coordinates["xtl"]), int(coordinates["ytl"]), int(coordinates["xbr"]), int(coordinates["ybr"])
                color = (255, 0, 0)  # Albastru pentru ieșirea vehiculului
                cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color, 2)
                cv2.putText(img, f"ID: {track_id}", (xtl, ytl - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Salvează imaginea de la ieșire
                output_exit = os.path.join(output_folder, f"vehicle_{track_id}_exit_frame_{frame}.png")
                cv2.imwrite(output_exit, img)

    return vehicle_tracks


def main():
    """
    Funcția principală.
    """
    # Setează căile către foldere și fișiere
    xml_path = "annotations.xml"
    images_folder = "images"
    output_folder = "output_tracking"

    # Parsează fișierul XML
    print("Parsez fișierul XML...")
    frame_data = parse_annotations(xml_path)

    # Numără vehiculele pe fiecare cadru
    print("Număr vehicule pe fiecare cadru...")
    frame_counts = count_vehicles(frame_data)
    for frame, counts in frame_counts.items():
        print(f"Cadru {frame}: {counts['cars']} mașini, {counts['minivans']} microbuze")

    # Tracking vehicule selectate
    selected_vehicle_ids = ["1", "2", "3"]  # Exemplu de ID-uri selectate pentru tracking
    print(f"Tracking vehicule selectate: {selected_vehicle_ids}")
    vehicle_tracks = track_selected_vehicles(frame_data, images_folder, output_folder, selected_vehicle_ids)

    # Rezumatul vehiculelor urmărite
    print("\nRezumat vehicule urmărite:")
    for track_id, info in vehicle_tracks.items():
        print(f"Vehicul ID {track_id}:")
        print(f"  Cadru intrare: {info['enter_frame']}")
        print(f"  Cadru ieșire: {info['exit_frame'] if info['exit_frame'] else 'nedefinit'}")


if __name__ == "__main__":
    main()
