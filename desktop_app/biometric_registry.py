import gradio as gr
import face_recognition
import numpy as np
import json
import os
from PIL import Image as PILImage 
from typing import Dict, Any, Optional

# --- Configuration: Database Path ---

DATABASE_DIR = os.environ.get("REGISTRY_DIR", os.path.join(os.path.expanduser("~"), "biometric_registry"))
IMAGES_DIR = os.path.join(DATABASE_DIR, "images") 
DATABASE_FILENAME = "face_data.json"
DATABASE_FILE = os.path.join(DATABASE_DIR, DATABASE_FILENAME)

TOLERANCE = 0.55 

# List of Cities for Dropdowns
SYRIAN_CITIES = [
    "Damascus", "Idleb", "Aleppo", "Homs", "Hama", "Latakia", 
    "Deir AL-Zor", "Tartous", "Dara'a", "Damascus Countryside", 
    "Raqa", "Al-Haska", "Swaida'a"
]

# --- Database & Core Functions ---

def load_database() -> Dict[str, Any]:
    """Loads the face database from the JSON file."""
    if os.path.exists(DATABASE_FILE):
        print(f"Loading database from: {DATABASE_FILE}")
        with open(DATABASE_FILE, 'r') as f:
            try:
                data = json.load(f)
                for person_id, person_data in data.items():
                    if isinstance(person_data, dict) and 'encoding' in person_data and isinstance(person_data['encoding'], list):
                        person_data['encoding'] = np.array(person_data['encoding'])
                return data
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {e}")
                return {}
            except Exception as e:
                 print(f"Error loading database: {e}")
                 return {}
    
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    return {}

def save_database(database: Dict[str, Any]):
    """Saves the face database to the JSON file."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    serializable_db = {}
    for person_id, person_data in database.items():
        person_data_copy = person_data.copy()
        if isinstance(person_data_copy.get('encoding'), np.ndarray):
            person_data_copy['encoding'] = person_data_copy['encoding'].tolist()
        
        serializable_db[person_id] = person_data_copy
        
    with open(DATABASE_FILE, 'w') as f:
        json.dump(serializable_db, f, indent=4)

def get_face_embedding(image_array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Detects face and returns embedding."""
    if image_array is None:
        return None
    try:
        # Ensure image is RGB (remove alpha channel if present) to prevent library errors
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
            
        face_encodings = face_recognition.face_encodings(image_array)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        print(f"Error processing image array: {e}")
        return None

def save_image_to_disk(image_array: np.ndarray, person_id: str, name: str) -> str:
    """
    Saves the numpy image array as a JPG file to the disk.
    Automatically converts PNG/RGBA to RGB to prevent errors.
    """
    try:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(' ', '_')
        filename = f"{person_id}_{safe_name}.jpg"
        file_path = os.path.join(IMAGES_DIR, filename)
        
        # Convert to PIL Image
        img = PILImage.fromarray(image_array)
        
        # CRITICAL FIX: Convert RGBA (transparent) to RGB (solid) before saving as JPG
        if img.mode in ("RGBA", "P"):
             img = img.convert("RGB")
             
        img.save(file_path, quality=95)
        return file_path
    except Exception as e:
        print(f"Error saving image file: {e}")
        return ""

# --- Gradio Action Functions ---

def add_new_person_gradio(
    img_array: Optional[np.ndarray], db_state: Dict[str, Any],
    name: str, mother_name: str, father_name: str, gender: str, 
    address: str, contact: str, mobile: str,
    id_card: str, city_birth: str, city_registered: str, notes: str
):
    # Default error return structure (must match output count: 1 state + 1 status + 1 image + 11 inputs = 14)
    empty_outputs = ("",) * 11 
    
    try:
        if name is None: name = ""
        
        if img_array is None or not name.strip():
             return (db_state, "ERROR: Please upload a photo and enter the full name.", None) + empty_outputs
        
        embedding = get_face_embedding(img_array)
        if embedding is None:
            return (db_state, "ERROR: No face detected. Please try a clearer photo.", None) + empty_outputs

        person_id = str(len(db_state) + 1).zfill(4)

        # Save image and get the permanent path
        saved_image_path = save_image_to_disk(img_array, person_id, name)

        person_document = {
            'id': person_id,
            'name': name.strip(),
            'mother_name': mother_name.strip() if mother_name else "",
            'father_name': father_name.strip() if father_name else "",
            'gender': gender,
            'address': address.strip() if address else "",
            'contact': contact.strip() if contact else "",
            'mobile': mobile.strip() if mobile else "",
            'id_card': id_card.strip() if id_card else "",
            'city_birth': city_birth,
            'city_registered': city_registered,
            'notes': notes.strip() if notes else "",
            'image_path': saved_image_path,
            'encoding': embedding 
        }
        
        db_state[person_id] = person_document
        save_database(db_state)
        
        success_msg = f"SUCCESS: Added '{name.strip()}' (ID: {person_id}). Image saved to disk."
        
        return (
            db_state, 
            success_msg, 
            None, # Clear image input
            "", "", "", "Male", "", "", "", "", None, None, "" # Clear text inputs & dropdowns (including notes)
        )
    except Exception as e:
        error_msg = f"CRITICAL SYSTEM ERROR: {str(e)}"
        print(error_msg)
        return (db_state, error_msg, None) + empty_outputs


def search_for_person_gradio(
    query_img_array: Optional[np.ndarray], 
    query_name: str, 
    query_city_registered: str, # Removed City of Birth input
    db_state: Dict[str, Any]
):
    """Searches by Face OR Name, with optional City of Registration filtering."""
    
    if not db_state:
        return "WARNING: Database is empty.", None, None

    # 1. Filter Database by City of Registration (if provided)
    filtered_db = db_state.copy()
    
    if query_city_registered:
        filtered_db = {pid: data for pid, data in filtered_db.items() if data.get('city_registered') == query_city_registered}
        
    if not filtered_db:
        return f"No records found matching the selected city criteria.", None, None

    # --- STRATEGY 1: Search by Image (Biometric) ---
    if query_img_array is not None:
        query_embedding = get_face_embedding(query_img_array)
        if query_embedding is None:
            return "SEARCH FAILED: No face detected in the uploaded photo.", None, None

        valid_records = [
            (pid, data) for pid, data in filtered_db.items() 
            if 'encoding' in data and isinstance(data['encoding'], np.ndarray)
        ]
        
        if not valid_records:
             return "WARNING: No valid biometric records found in the filtered list.", None, None

        known_ids = [r[0] for r in valid_records]
        known_encodings = [r[1]['encoding'] for r in valid_records]

        face_distances = face_recognition.face_distance(known_encodings, query_embedding)
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]
        
        if best_match_distance <= TOLERANCE:
            match_id = known_ids[best_match_index]
            person_data = filtered_db[match_id]
            return format_match_result(person_data, match_type="Biometric Scan", confidence=best_match_distance)
        else:
            closest_id = known_ids[best_match_index]
            closest_person = filtered_db[closest_id]
            return format_no_match_result(closest_person, closest_id, best_match_distance)

    # --- STRATEGY 2: Search by Name (Text) ---
    elif query_name and query_name.strip():
        search_term = query_name.lower().strip()
        found_records = []
        
        for pid, data in filtered_db.items():
            if search_term in data.get('name', '').lower():
                found_records.append(data)
        
        if found_records:
            person_data = found_records[0]
            match_note = f"Name Search (Found {len(found_records)} matches)"
            return format_match_result(person_data, match_type=match_note, confidence=0.0)
        else:
            return f"No records found matching name '{query_name}' in the selected cities.", "", None

    # --- STRATEGY 3: Search by City Filters Only ---
    elif query_city_registered and not query_name:
         # Return the first record that matches the city filters
         first_id = list(filtered_db.keys())[0]
         person_data = filtered_db[first_id]
         match_note = f"City Filter Match (Found {len(filtered_db)} total)"
         return format_match_result(person_data, match_type=match_note, confidence=0.0)

    else:
        return "ERROR: Please either upload a photo, enter a name, or select a city to search.", None, None

def search_for_delete_gradio(name: str, id_card: str, db_state: Dict[str, Any]):
    """Search specifically for deleting a record by Name or ID Card."""
    if not db_state:
        return "Database is empty.", "", None, None, gr.update(visible=False)
        
    if not name and not id_card:
        return "Please enter a Name or ID Card number.", "", None, None, gr.update(visible=False)

    found_id = None
    found_data = None

    # Priority 1: Search by ID Card (Exact match)
    if id_card:
        for pid, data in db_state.items():
            if data.get('id_card') == id_card.strip():
                found_id = pid
                found_data = data
                break
    
    # Priority 2: Search by Name (Partial match) if ID not found
    if not found_id and name:
        for pid, data in db_state.items():
            if name.lower() in data.get('name', '').lower():
                found_id = pid
                found_data = data
                break
    
    if found_id and found_data:
        # Prepare display details
        details = (
            f"ID: {found_data.get('id')}\n"
            f"Name: {found_data.get('name')}\n"
            f"ID Card: {found_data.get('id_card')}\n"
            f"City of Birth: {found_data.get('city_birth')}\n"
            f"City of Registration: {found_data.get('city_registered')}\n"
            f"Mother: {found_data.get('mother_name')}\n"
            f"Father: {found_data.get('father_name')}\n"
        )
        # Show delete button
        return f"Record Found: {found_data.get('name')}", details, found_data.get('image_path'), found_id, gr.update(visible=True)
    else:
        # Hide delete button
        return "No matching record found.", "", None, None, gr.update(visible=False)

def delete_record_gradio(person_id: str, db_state: Dict[str, Any]):
    """Deletes the record and its image file."""
    if person_id in db_state:
        data = db_state[person_id]
        image_path = data.get('image_path')
        name = data.get('name')
        
        # 1. Delete Image File
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Warning: Could not delete image file: {e}")
        
        # 2. Delete from Dictionary
        del db_state[person_id]
        
        # 3. Save Database
        save_database(db_state)
        
        return db_state, f"SUCCESS: Deleted record for {name} (ID: {person_id})", "", "", None, None, gr.update(visible=False), gr.update(visible=False)
    else:
        return db_state, "Error: Record ID not found in database.", "", "", None, None, gr.update(visible=False), gr.update(visible=False)

def format_match_result(person_data, match_type, confidence):
    detail_text = (
        f"--- MATCH FOUND ({match_type}) ---\n"
        f"ID: {person_data.get('id', 'N/A')}\n"
        f"Name: {person_data['name']}\n"
        f"ID Card: {person_data.get('id_card', 'N/A')}\n"
        f"City of Birth: {person_data.get('city_birth', 'N/A')}\n"
        f"City of Registration: {person_data.get('city_registered', 'N/A')}\n"
        f"Address: {person_data.get('address', '')}\n"
        f"Mother: {person_data.get('mother_name', '')}\n"
        f"Father: {person_data.get('father_name', '')}\n"
        f"Contact: {person_data.get('contact', '')} / Mobile: {person_data.get('mobile', '')}\n"
    )
    
    notes = person_data.get('notes', '')
    if notes:
        detail_text += f"\n--- NOTES ---\n{notes}\n"

    if confidence > 0:
        detail_text += f"\nConfidence: {confidence:.4f} (Lower is better)"
        
    status_msg = f"SUCCESS: Record found for {person_data['name']}"
    stored_image_path = person_data.get('image_path')
    return status_msg, detail_text, stored_image_path

def format_no_match_result(closest_person, closest_id, distance):
    detail_text = (
        f"--- NO EXACT MATCH ---\n"
        f"Closest record: {closest_person['name']} (ID: {closest_id})\n"
        f"City of Registration: {closest_person.get('city_registered', 'N/A')}\n"
        f"Distance: {distance:.4f} (Tolerance: {TOLERANCE})\n"
    )
    status_msg = "WARNING: No confirmed biometric match found."
    stored_image_path = closest_person.get('image_path')
    return status_msg, detail_text, stored_image_path

# --- Gradio Interface Setup ---

initial_db = load_database()

# Initialize Components
name_input = gr.Textbox(label="Full Name", placeholder="John Doe")
id_card_input = gr.Textbox(label="ID Card Number")
mother_input = gr.Textbox(label="Mother's Name")
father_input = gr.Textbox(label="Father's Name")
gender_input = gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender", value="Male")
address_input = gr.Textbox(label="Address")
contact_input = gr.Textbox(label="Contact Number")
mobile_input = gr.Textbox(label="Mobile Number")
city_birth_input = gr.Dropdown(choices=SYRIAN_CITIES, label="City of Birth")
city_registered_input = gr.Dropdown(choices=SYRIAN_CITIES, label="City of Registration")
notes_input = gr.Textbox(label="Additional Notes", lines=4, placeholder="Enter any other details here...")

add_image_input = gr.Image(type="numpy", label="Upload Photo to Register", height=300)

# Search Inputs
search_image_input = gr.Image(type="numpy", label="Option A: Search by Face", height=300)
search_name_input = gr.Textbox(label="Option B: Search by Name", placeholder="Enter name to find...")
search_city_registered_input = gr.Dropdown(choices=SYRIAN_CITIES, label="Filter by City of Registration")


search_status_output = gr.Textbox(label="Search Status", interactive=False)
match_details_output = gr.Textbox(label="Match Details", lines=12, interactive=False)
match_image_output = gr.Image(label="Stored Database Photo", height=300, interactive=False)

# Theme Setup
theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    radius_size=gr.themes.sizes.radius_md, 
).set(
    body_background_fill="#f7fafc",
    block_background_fill="#ffffff",
    button_primary_background_fill="#4c51bf",
    button_primary_border_color="#4c51bf",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=theme, title="Facial Biometric Database System") as demo:
    db_state = gr.State(initial_db)
    
    gr.Markdown("# 🧬 Facial Biometric Database GUI")
    gr.Markdown(f"**Storage Location:** `{DATABASE_DIR}`")
    
    current_record_count = gr.Textbox(
        label="Total Records", 
        value=f"{len(initial_db)}", 
        interactive=False, 
        container=True
    )
    
    global_status = gr.Textbox(label="System Status", interactive=False, container=True)

    with gr.Tabs():
        # TAB 1: ADD PERSON
        with gr.TabItem("Add New Person"):
            with gr.Row():
                with gr.Column(scale=1):
                    add_image_input.render()
                with gr.Column(scale=2):
                    gr.Markdown("## Record Details")
                    name_input.render()
                    id_card_input.render()
                    
                    with gr.Row():
                        city_birth_input.render()
                        city_registered_input.render()

                    with gr.Row():
                        mother_input.render()
                        father_input.render()
                    
                    address_input.render()
                    
                    with gr.Row():
                        contact_input.render()
                        mobile_input.render()
                    gender_input.render()
                    notes_input.render()
                    
            add_button = gr.Button("Add Person & Save Image", variant="primary")
            
            add_button.click(
                fn=add_new_person_gradio,
                inputs=[
                    add_image_input, db_state, name_input, mother_input, 
                    father_input, gender_input, address_input, contact_input, mobile_input,
                    id_card_input, city_birth_input, city_registered_input, notes_input
                ],
                outputs=[
                    db_state, global_status, add_image_input, name_input, mother_input, 
                    father_input, gender_input, address_input, contact_input, mobile_input,
                    id_card_input, city_birth_input, city_registered_input, notes_input
                ]
            ).success(
                lambda db: str(len(db)),
                inputs=[db_state],
                outputs=[current_record_count]
            )
        
        # TAB 2: SEARCH PERSON
        with gr.TabItem("Search Person"):
            gr.Markdown("## Search Database")
            gr.Markdown("Use any combination of Face, Name, or City Filters.")
            
            with gr.Row():
                with gr.Column():
                    search_image_input.render()
                    search_name_input.render()
                    with gr.Row():
                        search_city_registered_input.render()
                    search_button = gr.Button("Search Database", variant="primary")
                
                with gr.Column():
                    search_status_output.render()
                    match_details_output.render()
            
            match_image_output.render()
            
            search_button.click(
                fn=search_for_person_gradio,
                inputs=[
                    search_image_input, search_name_input, 
                    search_city_registered_input, 
                    db_state
                ],
                outputs=[search_status_output, match_details_output, match_image_output]
            )

        # TAB 3: DELETE PERSON
        with gr.TabItem("Delete Person"):
            gr.Markdown("## Delete Record")
            gr.Markdown("Search by **Name** or **ID Card** to find a record to delete.")
            
            with gr.Row():
                delete_name_input = gr.Textbox(label="Search by Name")
                delete_id_input = gr.Textbox(label="Search by ID Card")
            
            find_delete_btn = gr.Button("Find Record")
            
            # Hidden state to store the ID of the person found for deletion
            target_delete_id = gr.State()
            
            gr.Markdown("### Record Preview")
            with gr.Row():
                delete_status = gr.Textbox(label="Status", interactive=False)
                delete_details = gr.Textbox(label="Record Details", lines=7, interactive=False)
                delete_image = gr.Image(label="Record Photo", height=250, interactive=False)
            
            # Section for deletion buttons
            initiate_delete_btn = gr.Button("Delete This Record", variant="stop", visible=False)
            
            # Confirmation Group (Hidden initially)
            with gr.Group(visible=False) as confirm_group:
                gr.Markdown("### ⚠️ Are you sure you want to delete this record? This action cannot be undone.")
                with gr.Row():
                    confirm_delete_btn = gr.Button("Yes, Permanently Delete", variant="stop")
                    cancel_delete_btn = gr.Button("No, Cancel")

            # 1. Find Record Action
            find_delete_btn.click(
                fn=search_for_delete_gradio,
                inputs=[delete_name_input, delete_id_input, db_state],
                outputs=[delete_status, delete_details, delete_image, target_delete_id, initiate_delete_btn]
            )
            
            # 2. Initiate Delete (Show Confirmation)
            initiate_delete_btn.click(
                fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[initiate_delete_btn, confirm_group]
            )

            # 3. Confirm Delete Action
            confirm_delete_btn.click(
                fn=delete_record_gradio,
                inputs=[target_delete_id, db_state],
                outputs=[
                    db_state, global_status, delete_name_input, delete_id_input, 
                    delete_details, delete_image, initiate_delete_btn, confirm_group
                ]
            ).success(
                lambda db: str(len(db)),
                inputs=[db_state],
                outputs=[current_record_count]
            )
            
            # 4. Cancel Delete Action
            cancel_delete_btn.click(
                fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                outputs=[initiate_delete_btn, confirm_group]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_api=False)