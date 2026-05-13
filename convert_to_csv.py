import csv

input_file = "dataset/captions.txt"
output_file = "dataset/dataset.csv"

with open(input_file, "r", encoding="utf-8") as txt_file:
    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(["image", "caption"])

        for line in txt_file:
            parts = line.strip().split("\t")

            if len(parts) != 2:
                continue

            image, caption = parts

            writer.writerow([image, caption])

print("✅ تم تحويل الملف إلى CSV")