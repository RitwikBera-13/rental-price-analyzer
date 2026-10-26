import shutil, os

os.makedirs("static/img", exist_ok=True)

shutil.copy("/mnt/data/Screenshot 2025-11-21 012908.png", "static/img/skyline.png")
shutil.copy("/mnt/data/Screenshot 2025-11-21 013047.png", "static/img/livingroom.png")

print("Copied skyline.png and livingroom.png into static/img/")
