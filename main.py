import face_recognition
import cv2
import csv

faces = []
names = []
scanned_faces = []  

face1img = face_recognition.load_image_file("/Users/tr/Desktop/APFTSM/ritikamaam.jpg")
face2img = face_recognition.load_image_file("/Users/tr/Desktop/APFTSM/mudit.jpg")
face3img = face_recognition.load_image_file("/Users/tr/Desktop/APFTSM/teisha.jpg")
face4img = face_recognition.load_image_file("/Users/tr/Desktop/APFTSM/sam.jpg")

face1 = face_recognition.face_encodings(face1img)[0]
face2 = face_recognition.face_encodings(face2img)[0]
face3 = face_recognition.face_encodings(face3img)[0]
face4 = face_recognition.face_encodings(face4img)[0]

faces.append(face1)
faces.append(face2)
faces.append(face3)
faces.append(face4)

names.append("ritika ma'am")
names.append("mudit")
names.append("teisha")
names.append("samaira")

vc = cv2.VideoCapture(0)
while True:
    ret, frame = vc.read()
    floca = face_recognition.face_locations(frame)
    fenco = face_recognition.face_encodings(frame, floca)
    for (top, right, bottom, left), face_encoding in zip(floca, fenco):
        matches = face_recognition.compare_faces(faces, face_encoding)
        name = "unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]
            if name not in scanned_faces:
                scanned_faces.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 225), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 225), 2)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

with open("attendance.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name"]) 
    for face in scanned_faces:
        writer.writerow([face])  

print("Scanned Faces:", scanned_faces)
vc.release()
cv2.destroyAllWindows()
