#عشان تبعت requests لـ Unsplash API وتحمل الصور
import requests
#عشان تعمل فولدر الصور وتحفظ الملفات في path معين
import os
#عشان تعرض progress bar أثناء تحميل الصور
from tqdm import tqdm
#عشان تعمل delay بين الصفحات وتتجنب الـ rate limit أو ban
import time

ACCESS_KEY = "0kW5lpCyiJKfXDc1HS1kFsw4lPHoKOpy7zmkP8xQ4GY"

IMAGE_DIR = "dataset/images"
CAPTION_FILE = "dataset/captions.txt"
#يعمل فولدر الصور لو مش موجود.
os.makedirs(IMAGE_DIR, exist_ok=True)

queries = ["food", "sports"]

def download_images():
    count = 0

    with open(CAPTION_FILE, "a", encoding="utf-8") as f:

        for query in queries:
            print(f"🔍 Searching for: {query}")

            for page in range(1, 50):  # عدد الصفحات

                url = "https://api.unsplash.com/search/photos" #أنت بتبعت request للـ endpoint ده:

                params = {
                    "query": query,
                    "per_page": 30,
                    "page": page,
                    "client_id": ACCESS_KEY
                }
                #وده بيبعت الطلب لـ Unsplash ويرجعلك response.
                response = requests.get(url, params=params)

                if response.status_code != 200:
                    print("❌ وقفنا بسبب limit أو error")
                    return

                data = response.json()

                #بتلف على كل صورة: عشان اخد الصورة نفسها والوصف 
                for photo in tqdm(data["results"]):
                    img_url = photo["urls"]["small"]
                    caption = photo["alt_description"]
                    #لو مفيش caption، بتعمل skip للصورة:
                    if caption is None:
                        continue
                    # بسمي الصورة
                    img_name = f"{query}_{count}.jpg"
                    img_path = os.path.join(IMAGE_DIR, img_name)


                    #هنا أنت بتعمل request لرابط الصورة نفسه.
                    #بيدخل على الرابط ويحمل محتوى الصورة كـ binary data.
                    #هنا بتفتح ملف جديد عشان تحفظ الصورة فيه.
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    #حفظ الوصف 
                    f.write(f"{img_name}\t{caption}\n")
                     


# ده بيزود عداد الصور بعد ما كل صورة تتحمل وتتسجل يعني يخلي الاسم جمبه 0 تحته 1 تحته 2 
                    count += 1
            

            #ده بيوقف البرنامج ثانية واحدة بعد كل صفحة من نتائج Unsplash.
                time.sleep(1)  # عشان منعملش ban

    print(f"✅ تم تحميل {count} صورة")


download_images()