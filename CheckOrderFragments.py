import os 

LA_path = "/Users/hagedorn/Desktop/YOLO/LA"
RA_path = "/Users/hagedorn/Desktop/YOLO/RA"

for la, ra in zip(sorted(os.listdir(LA_path)),sorted(os.listdir(RA_path))): 
    if la[3:] != ra[3:]: 
        print(f"Not Correct! LA: {la}, RA: {ra}")
    else: 
        print("Correct!")