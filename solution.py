import csv
import cv2
from ultralytics import YOLO

# Load the video and the model
cap = cv2.VideoCapture("video1_Trim.mp4")
model = YOLO("yolov8n.pt")

# Create and open the CSV file in write mode
csv_file = open('object_durations.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['ID', 'Duration (s)', 'Entry Time', 'Exit Time'])  # Write header row to CSV

# Get the video fps
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize an empty dictionary to store the object times
object_times = {}

# Loop through the video frames
while True:
    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Perform object detection
    results = model.track(frame, persist=True, classes=[0])

    # Get the current frame index
    current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

    # Calculate the current time in seconds
    current_time = (current_frame_idx - 1) / fps
    
    # Get the bounding boxes and IDs for detected objects
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Loop through the bounding boxes and IDs
    for box, obj_id in zip(boxes, ids):
        # Check if the object is new or already tracked
        if obj_id not in object_times:
            object_times[obj_id] = {
                'entry_time': current_time,
                'exit_time': None, # Initialize exit time to None
                'total_duration': 0  # Initialize total duration to 0
            }

        # Display the bounding box on the frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Display the time counter on the frame
        cv2.putText(
            frame,
            f"ID:{obj_id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        # Get the entry time for the object
        object_entry_time = object_times[obj_id]['entry_time']

        # Calculate the time counter for the object
        time_counter = current_time - object_entry_time

        # Display the time counter on the frame
        cv2.putText(
            frame,
            f"Time: {time_counter:.2f}s",
            (box[0], box[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        # Update the total duration for the object
        object_times[obj_id]['total_duration'] = time_counter
        
        # Update the exit time for the object
        object_times[obj_id]['exit_time'] = current_time

    # Display the frame
    cv2.imshow("frame", frame)

    # Press `q` to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Loop through the object times dictionary and write the data to CSV
for obj_id, obj_data in object_times.items():    
    entry_time = obj_data['entry_time']
    exit_time = obj_data['exit_time']
    total_duration = obj_data['total_duration']

    csv_writer.writerow([obj_id, total_duration, entry_time, exit_time])

# Release the video capture object and close the CSV file
csv_file.close()
cap.release()
cv2.destroyAllWindows()
