from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10b')
source = '/mnt/HDD1/phudh/course/calar/IDS_s24/HW0/results/DOS/TCP/DOS_01_town05_04_18_00_09_29/rgb/0023.png'
model.predict(source=source, save=True)
