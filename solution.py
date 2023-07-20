import cv2
from ultralytics import YOLO
import supervision as sv
import time
import csv

# Load the video and the model
cap = cv2.VideoCapture("video1_Trim.mp4")
model = YOLO("yolov8n.pt")

object_times = {}

# Create and open the CSV file in write mode
csv_file = open('object_durations.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['ID', 'Duration (s)'])  # Write header row to CSV

# Loop through the video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(frame, persist=True)

    model.classes = [0]

    # Get the current time in seconds
    current_time = time.time()  

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)

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
            0.75,
            (0, 0, 255),
            1,
        )

        object_entry_time = object_times[obj_id]['entry_time']

        time_counter = current_time - object_entry_time

        # Display the time counter on the frame
        # cv2.putText(
        #     frame,
        #     f"Time: {time_counter:.2f}s",
        #     (box[0], box[1] - 20),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 255),
        #     2,
        # )

        # Update the total duration for the object
        object_times[obj_id]['total_duration'] = time_counter

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Update the exit time of objects that are no longer in the frame
    tracked_ids = set(ids)
    remove_ids = []

    for obj_id in object_times.keys():
        if obj_id not in tracked_ids:
            object_times[obj_id]['exit_time'] = current_time

            # Retrieve the total duration for the object
            total_duration = object_times[obj_id]['total_duration']

            # Write the ID and total duration to the CSV file
            csv_writer.writerow([obj_id, total_duration])

            # Add the ID to the remove_ids list
            remove_ids.append(obj_id)

    # Remove the objects from the dictionary
    for obj_id in remove_ids:
        del object_times[obj_id]

# Close the CSV file
csv_file.close()
