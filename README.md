# Hear My Hands – ASL Recognition

## Run the Application

Follow these steps to run the project on your computer.

### 1. Install Python, dataset and csv files used from the links given below

Install Python 3.10 or a compatible version supported by the project's MediaPipe installation.

dataset: 1. Alphabets: https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download
	 
	 2. Words: https://www.kaggle.com/datasets/shubhambhardwaj01/american-sign-language-image-dataset?resource=download

csv files zip: https://drive.google.com/file/d/1R9ESGFwpy9teZtjIl9mz64glplJMuwpt/view?usp=sharing

Check that Python and pip are available:

```bash
python --version
python -m pip --version
```

### 2. Clone the GitHub Repository

### 3. Install the Required Libraries

Install the project dependencies:

```bash
python -m pip install opencv-python mediapipe numpy pandas scikit-learn joblib matplotlib seaborn pillow
```

### 4. Check the Project Files

Make sure the repository contains the required application files:

- `app.py`
- `lr_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `hand_landmarker.task`
- The background image referenced in `app.py` (for example, `bg2.jpg`)

Keep these files in the locations expected by the code. If the background image has a different filename, update its filename in `app.py`.

### 5. Connect Your Webcam

Connect or enable your computer's webcam. Close other applications that may be using it.

### 6. Run the Application

From the project directory, execute:

```bash
python app.py
```

The Hear My Hands window should open and display the webcam feed and recognition dashboard. Show one hand clearly in front of the camera.

### 7. Use the Keyboard Controls

| Key | Action |
|---|---|
| `ENTER` | Add the currently predicted sign to the message |
| `SPACE` | Delete the last character from the message |
| `C` | Clear the message |
| `V` | Call the message/speech function |
| `Q` | Quit the application |

---

**Project:** Hear My Hands – ASL Alphabet Recognition  
**Developer:** Priyansh Rawat  
**Enrollment No.:** S25CSEU0829  
**Faculty:** Dr Basab Nath
