from flask import Flask, render_template, request, redirect, url_for, send_from_directory
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

# Home page route to upload video
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            # Call the analysis function here
            analyze_video(video_path)

            return redirect(url_for('results'))
    return redirect(url_for('index'))

# Function to analyze the video (Pose estimation and plotting)
def analyze_video(video_path):
    # Initialize Mediapipe pose estimator
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    # Set video parameters
    joint_a = "14"
    joint_b = "12"
    joint_c = "24"

    # File paths
    armor_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'armor_output.csv')
    plot_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ROM_angle_plot.png')

    # Video capture and pose analysis logic
    cap = cv2.VideoCapture(video_path)
    phrases_dict = {str(i): [] for i in range(1, 33)}
    frame_data = []  # To store landmark data

    frame_count = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Resize the frame to 50% of its original size
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        # Process every 2nd frame for efficiency
        if frame_count % 2 == 0:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                # Collect landmark data
                landmarks = [lm for lm in results.pose_landmarks.landmark]
                frame_data.append(landmarks)

        frame_count += 1

    cap.release()

    # Process captured data into CSV
    if frame_data:
        max_frames = len(frame_data)
        num_landmarks = len(frame_data[0])

        # Create a DataFrame for all frames and landmarks
        landmark_array = np.zeros((max_frames, num_landmarks * 3))  # 3 values (x, y, z) per landmark

        for i, landmarks in enumerate(frame_data):
            for j, lm in enumerate(landmarks):
                landmark_array[i, j * 3] = lm.x
                landmark_array[i, j * 3 + 1] = lm.y
                landmark_array[i, j * 3 + 2] = lm.z

        # Convert to DataFrame
        landmark_df = pd.DataFrame(landmark_array, columns=[f"x{j}" for j in range(num_landmarks)] + 
                                                          [f"y{j}" for j in range(num_landmarks)] + 
                                                          [f"z{j}" for j in range(num_landmarks)])

        # Calculate angles using vectorized operations
        armor_output = []
        for i in range(max_frames):
            joint_a_x, joint_b_x, joint_c_x = landmark_df.loc[i, [f"x{joint_a}", f"x{joint_b}", f"x{joint_c}"]]
            joint_a_y, joint_b_y, joint_c_y = landmark_df.loc[i, [f"y{joint_a}", f"y{joint_b}", f"y{joint_c}"]]

            BA = np.array([joint_a_x - joint_b_x, joint_a_y - joint_b_y])
            BC = np.array([joint_c_x - joint_b_x, joint_c_y - joint_b_y])
            dot_product = np.dot(BA, BC)
            magnitude_BA = np.linalg.norm(BA)
            magnitude_BC = np.linalg.norm(BC)

            if magnitude_BA > 0 and magnitude_BC > 0:
                theta_radians = math.acos(dot_product / (magnitude_BA * magnitude_BC))
                theta_degrees = math.degrees(theta_radians)
            else:
                theta_degrees = None

            armor_output.append([i + 1, theta_degrees])

        task1_angle_df = pd.DataFrame(armor_output, columns=["Time", "Degrees"])
        task1_angle_df.to_csv(armor_output_path, index=False)

        # Plot and save the angle data
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=task1_angle_df, x="Time", y="Degrees")
        plt.title("Range of Motion Quantification")
        plt.xlabel("Frame")
        plt.ylabel("Angle of Interest (Degrees)")
        sns.set_style("whitegrid")
        plt.savefig(plot_output_path)


# Route to display the results
@app.route('/results')
def results():
    return render_template('results.html')

# Route to serve the CSV file for download
@app.route('/download_csv')
def download_csv():
    csv_filename = 'armor_output.csv'
    directory = os.path.join(app.root_path, 'static', 'uploads')
    return send_from_directory(directory=directory, path=csv_filename, as_attachment=True)

if __name__ == "__main__":
    #app.run(debug=True)
    app.run()
