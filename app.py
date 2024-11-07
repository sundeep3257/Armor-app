from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import mediapipe as mp
import time
import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Define paths for saving files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Joint values dictionary
JOINT_VALUES = {
    'right_elbow': {'a': '16', 'b': '14', 'c': '12'},
    'left_elbow': {'a': '15', 'b': '13', 'c': '11'},
    'right_shoulder': {'a': '14', 'b': '12', 'c': '24'},
    'left_shoulder': {'a': '13', 'b': '11', 'c': '23'},
    'right_knee': {'a': '24', 'b': '26', 'c': '28'},
    'left_knee': {'a': '23', 'b': '25', 'c': '27'}
}

selected_joints = None

# Home page route to upload video
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_joint', methods=['POST'])
def select_joint():
    global selected_joints
    joint = request.form.get('joint')
    if joint and joint in JOINT_VALUES:
        selected_joints = JOINT_VALUES[joint]
    return jsonify({"success": True, "selected_joint": selected_joints})

@app.route('/user_guide')
def user_guide():
    return render_template('userguide.html')

# Route to handle video upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['video']
        if selected_joints:
            joint_a = selected_joints['a']
            joint_b = selected_joints['b']
            joint_c = selected_joints['c']
        else:
            joint_a, joint_b, joint_c = '14', '12', '24'  # Default values

        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            # Call the analysis function with custom joints
            analyze_video(video_path, joint_a, joint_b, joint_c)

            return redirect(url_for('results'))
    return redirect(url_for('index'))

# Route to save recorded video
@app.route('/record', methods=['POST'])
def record():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_video.webm')
            file.save(video_path)
            return jsonify({"video_path": video_path}), 200
    return jsonify({"error": "Failed to record video"}), 400

# Function to analyze the video (Pose estimation and plotting)
def analyze_video(video_path, joint_a, joint_b, joint_c):
    # Initialize Mediapipe pose estimator
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    # File paths
    armor_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'armor_output.csv')
    plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ROM_angle_plot.png')

    # Video capture and pose analysis logic
    cap = cv2.VideoCapture(video_path)
    phrases_dict = {str(i): [] for i in range(1, 33)}

    # Determine frame skip rate
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip_rate = math.ceil(original_fps / target_fps)

    frame_count = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Resize the frame to 25% of its original size
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        # Process every 2nd frame for efficiency
        if frame_count % frame_skip_rate == 0:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                # Extract landmarks in a single loop
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    phrase = f"{id} x: {lm.x:.8f} y: {lm.y:.8f} z: {lm.z:.8f} visibility: {lm.visibility:.8f}"
                    start_number = phrase.split(' ')[0]
                    if start_number in phrases_dict:
                        phrases_dict[start_number].append(phrase)

        frame_count += 1

    cap.release()

    # Process captured data into CSV
    if any(len(phrases) > 0 for phrases in phrases_dict.values()):
        max_instances = max(len(phrases) for phrases in phrases_dict.values())
        table_data = [['' for _ in range(32)] for _ in range(max_instances)]

        for row in range(32):
            str_row = str(row + 1)
            if str_row in phrases_dict:
                for i, phrase in enumerate(phrases_dict[str_row]):
                    if i < max_instances:
                        table_data[i][row] = phrase

        # Convert table_data to a DataFrame
        raw_data = pd.DataFrame(table_data)

        formatted_output = pd.DataFrame(" ", index=np.arange(len(raw_data)), columns=[f"x{i}" for i in range(1, 33)] + [f"y{i}" for i in range(1, 33)])

        for j in range(raw_data.shape[1]):
            for i in range(raw_data.shape[0]):
                raw_cell = str(raw_data.iloc[i, j]).split(" ")
                if len(raw_cell) >= 6:
                    number = int(raw_cell[0])
                    x_value = float(raw_cell[2])
                    y_value = float(raw_cell[4])
                    row_to_fill = formatted_output[formatted_output[f"x{number}"] == " "].index[0]
                    formatted_output.at[row_to_fill, f"x{number}"] = x_value
                    formatted_output.at[row_to_fill, f"y{number}"] = y_value

        # Calculate angles and save to CSV
        armor_output = []
        formatted_output.fillna(0, inplace=True)  # Replace NaNs with 0 for calculations
        for i in range(len(formatted_output)):
            joint_a_x, joint_b_x, joint_c_x = formatted_output.loc[i, [f"x{joint_a}", f"x{joint_b}", f"x{joint_c}"]].astype(float)
            joint_a_y, joint_b_y, joint_c_y = formatted_output.loc[i, [f"y{joint_a}", f"y{joint_b}", f"y{joint_c}"]].astype(float)

            BAx, BAy = joint_a_x - joint_b_x, joint_a_y - joint_b_y
            BCx, BCy = joint_c_x - joint_b_x, joint_c_y - joint_b_y
            dot_product = BAx * BCx + BAy * BCy
            magnitude_BA = math.sqrt(BAx ** 2 + BAy ** 2)
            magnitude_BC = math.sqrt(BCx ** 2 + BCy ** 2)

            # Avoid division by zero
            theta_degrees = None
            if magnitude_BA > 0 and magnitude_BC > 0:
                theta_radians = math.acos(dot_product / (magnitude_BA * magnitude_BC))
                theta_degrees = math.degrees(theta_radians)

            armor_output.append([i + 1, theta_degrees])

        task1_angle_df = pd.DataFrame(armor_output, columns=["Time", "Degrees"])
        task1_angle_df.to_csv(armor_output_path, index=False)

        # Plot and save the angle data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=task1_angle_df, x="Time", y="Degrees", color="#750014ff")  # Set the color here
        plt.title("Range of Motion Quantification")
        plt.xlabel("Frame")
        plt.ylabel("Angle of Interest (Degrees)")
        sns.set_style("whitegrid")
        plt.savefig(plot_output_path)

# Route to display the results
@app.route('/results')
def results():
    # Path to the CSV file
    armor_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'armor_output.csv')
    # Load the CSV into a DataFrame
    task1_angle_df = pd.read_csv(armor_output_path)
    # Find the maximum value in the 'Degrees' column (2nd column)
    peak_angle = round(task1_angle_df['Degrees'].max(), 2)
    # Find the minimum value in the 'Degrees' column (2nd column)
    min_angle = round(task1_angle_df['Degrees'].min(), 2)

    return render_template('results.html', peak_angle=peak_angle, min_angle=min_angle)

# Route to serve the CSV file for download
@app.route('/download_csv')
def download_csv():
    csv_filename = 'armor_output.csv'
    directory = os.path.join(app.root_path, 'static', 'uploads')
    return send_from_directory(directory=directory, path=csv_filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
    #app.run()
