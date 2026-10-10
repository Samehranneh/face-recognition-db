# Face Recognition Database

Register a person with a photo, then find them again later by searching with a new picture.

Each photo is turned into a 128-dimensional face embedding with the [`face_recognition`](https://github.com/ageitgey/face_recognition) library (dlib under the hood). Embeddings and details are stored in a JSON file that works like a small document store. A search compares the query embedding against every stored one and returns the closest person, with the distance so you can judge confidence.

## Why

In field work, the same person is often registered several times by different teams or at different sites. Matching by face helps flag likely duplicates before assistance is distributed. This repository is a working prototype of that idea.

## Setup

```bash
pip install -r requirements.txt
```

`face_recognition` needs dlib, which needs CMake and a C++ compiler. On Google Colab it installs with `pip` directly.

## Usage

```bash
# register
python face_db.py add sample_data/person1.jpg --name "Test Person" --note "site A"

# search
python face_db.py search sample_data/query.jpg

# list everyone
python face_db.py list
```

Example output:

```
Match: Test Person [0001], distance 0.3812 (tolerance 0.55)
```

Lower distance means more similar. Change `TOLERANCE` in `face_db.py` to make matching stricter or looser.

## Desktop app

`desktop_app/biometric_registry.py` is a full graphical version built with [Gradio](https://gradio.app):

- **Register** a person with a photo and case details (name, parents' names, ID card, address, phone, city of birth and registration, notes). The photo is saved and its face embedding stored.
- **Search** by photo (face match with a confidence score, or the closest record when nothing passes the threshold), by name, or by registration city.
- **Delete** a record after looking it up by name or ID card.

Records are stored in `~/biometric_registry` by default; set `REGISTRY_DIR` to change it.

```bash
pip install gradio face_recognition pillow numpy
python desktop_app/biometric_registry.py
```

`build_exe.bat` packages it as a single Windows `.exe` with PyInstaller.

## Data protection

Face embeddings are biometric data. Before using anything like this with real people:

- get informed consent and explain how the data will be used,
- store the database encrypted and with restricted access,
- keep as few personal fields as possible,
- follow your organization's data protection policy (and, in humanitarian settings, the ICRC handbook on data protection).

`face_data.json`, the app's data folder and saved photos are git-ignored so records are never committed. No real records or photos are included in this repository.

## Limitations

- One face per photo is assumed; the first face found is used.
- Search is a linear scan, fine for thousands of records but not millions.
- Accuracy drops with poor lighting, side angles, and children's faces.

## License

MIT
