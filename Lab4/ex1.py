import re

from pytesseract import pytesseract

class OCRProcessor:
    def __init__(self):
        pass

    def perform_ocr(self, image):

        ocr_result = pytesseract.image_to_string(image)

        output_file = "results/ocr_result.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ocr_result)
        #print(f"OCR text saved to {output_file}")

        return ocr_result

    def get_accuracy(self, ground_thruth_path):
        with open(ground_thruth_path, "r", encoding="utf-8") as f1, open("results/ocr_result.txt", "r", encoding="utf-8") as f2:
            text1 = re.sub(r"\s+", "", f1.read()).lower()

            text2_raw = f2.read()
            text2 = re.sub(r"\s+", "", text2_raw).lower()

            text2_no_first = text2_raw[1:]
            text2_var = re.sub(r"\s+", "", text2_no_first).lower()


        max_length = max(len(text1), len(text2))
        max_length_var = max(len(text1), len(text2_var))

        text1 = text1.ljust(max_length)
        text2 = text2.ljust(max_length)

        text1_var = text1.ljust(max_length_var)
        text2_var = text2_var.ljust(max_length_var)

        # comparare caractere
        identical_count = 0
        different_count = 0

        for char1, char2 in zip(text1, text2):
            if char1 == char2:
                identical_count += 1
            else:
                different_count += 1

        accuracy = identical_count/(different_count + identical_count)

        # text2 = text2[1:]
        identical_count_var = 0
        different_count_var = 0

        for char1, char2 in zip(text1_var, text2_var):
            if char1 == char2:
                identical_count_var += 1
            else:
                different_count_var += 1

        accuracy_var = identical_count_var / (different_count_var + identical_count_var)

        if accuracy > accuracy_var:
            return accuracy
        else:
            return accuracy_var
