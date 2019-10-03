Facial recognition using Python


** Instructions to use: **

Note: Creation of Virtual Environment is preferred.

After creating virtual environment, install required libraries using command:
pip install -r requirements.txt


Project structure:-
To see project structure,use:- $tree


Files explaination:

- collectedfaces/ : Contains output(images) from collect_faces.py.

- datasets/ : Contains face images from folder collectedfaces for people organized into subdirectories based on their respective names.

- output/ : This is where output(processed face recognition videos) gets stored.

- pickle_encodings/ : For storing encoded pickle files.Facial recognitions encodings are generated from your dataset via face_encoding.py

- collect_faces.py : This file is for collecting faces of different persons.

- face_encoding.py : Encodings (128-d vectors) for faces are built with this script.

- face_recognize_webcam_vdo.py : Recognize faces in a live video stream from your webcam and output a video.


<!-- After a dataset of images is created, we’ll run face_encoding.py.py  to build the embeddings. -->



Stepwise execution of commands:-

1. *** COLLECT + UPLOAD FACES *** (For collecting faces)

python3 collectfaces.py --encodings pickle_encodings/brmu.pickle --output output/webcam_face.avi --display 1 --detection-method hog

NOTE:- Collect each person's image on one by one basis.Put all collected images of first person in a folder named by his own name.Then collect another person's images and put it in another folder named by his name and so on.And finally preserve all these folders as subfolders under datasets folder.


2. *** TRAIN + DOWNLOAD MODEL(On Xtage Server) ***

python face_encoding.py --dataset datasets --encodings brmu.pickle

NOTE:- "We can give any name to pickle file while training the model.Here we have named it as brmu."
Then download brmu.pickle file.Store this downloaded file from server to pickle_encodings folder.


3. *** RUN RECOGNITION *** (For recognizing faces)

python3 face_recognize_webcam_vdo.py --encodings pickle_encodings/brmu.pickle --output output/webcam_face.avi --display 1 --detection-method hog



NOTE:- Use brmu.pickle file for this project.Otherwise, train the model and download a fresh pickle file from xtage server.











