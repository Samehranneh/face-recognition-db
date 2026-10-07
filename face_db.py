"""Face Recognition Database System.

Store a person's details together with a 128-dimensional face embedding,
then search the database with a new photo to find the closest match.

Records are kept in a JSON file that behaves like a small document store.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import face_recognition
import numpy as np

DATABASE_FILE = os.environ.get("FACE_DB_FILE", "face_data.json")
# Lower tolerance = stricter matching. 0.6 is the library default.
TOLERANCE = 0.55

Record = Dict[str, Any]


# ---------- storage ----------

def load_database(path: str = DATABASE_FILE) -> Dict[str, Record]:
    """Load records from disk, converting stored encodings back to numpy arrays."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Could not read {path}; starting with an empty database.")
        return {}
    for record in data.values():
        record["encoding"] = np.array(record["encoding"])
    return data


def save_database(db: Dict[str, Record], path: str = DATABASE_FILE) -> None:
    """Write records to disk, converting numpy encodings to lists for JSON."""
    serializable = {
        pid: {**rec, "encoding": np.asarray(rec["encoding"]).tolist()}
        for pid, rec in db.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


# ---------- recognition ----------

def get_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """Return the embedding of the first face found in the image, or None."""
    try:
        image = face_recognition.load_image_file(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        print(f"No face detected in {image_path}")
        return None
    return encodings[0]


def add_person(db: Dict[str, Record], image_path: str, details: Dict[str, str]) -> Optional[str]:
    """Add a person to the database. Returns the new ID, or None if no face was found."""
    embedding = get_face_embedding(image_path)
    if embedding is None:
        return None
    person_id = str(len(db) + 1).zfill(4)
    db[person_id] = {**details, "image_path": image_path, "encoding": embedding}
    save_database(db)
    return person_id


def search_person(db: Dict[str, Record], image_path: str) -> Optional[tuple[str, float, bool]]:
    """Find the closest stored face.

    Returns (person_id, distance, is_match) or None if the query has no face
    or the database is empty.
    """
    query = get_face_embedding(image_path)
    if query is None or not db:
        return None
    ids = list(db.keys())
    distances = face_recognition.face_distance([db[i]["encoding"] for i in ids], query)
    best = int(np.argmin(distances))
    return ids[best], float(distances[best]), bool(distances[best] <= TOLERANCE)


# ---------- command line ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition database")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="register a person from a photo")
    add.add_argument("image")
    add.add_argument("--name", required=True)
    add.add_argument("--note", default="", help="free-text note, e.g. case reference")

    find = sub.add_parser("search", help="find a person by photo")
    find.add_argument("image")

    sub.add_parser("list", help="list registered people")

    args = parser.parse_args()
    db = load_database()

    if args.command == "add":
        pid = add_person(db, args.image, {"name": args.name, "note": args.note})
        print(f"Added {args.name} as {pid}" if pid else "Nothing added.")
    elif args.command == "search":
        result = search_person(db, args.image)
        if result is None:
            print("No result: empty database or no face in the query image.")
            return
        pid, dist, ok = result
        verdict = "Match" if ok else "No match (closest shown)"
        print(f"{verdict}: {db[pid]['name']} [{pid}], distance {dist:.4f} (tolerance {TOLERANCE})")
    elif args.command == "list":
        for pid, rec in db.items():
            print(pid, rec["name"], rec.get("note", ""))


if __name__ == "__main__":
    main()
