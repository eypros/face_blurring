# combination of face blurring approach with opencv tracking
# working for multiple bboxes and also saves to disk
# examples found in https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
from random import randint

import os
import cv2
import sys

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def show(im):
    cv2.imshow("image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def init_video_recorder(video_output_path, camera):
    if os.path.isfile(video_output_path):
        video_name, ext = os.path.splitext(os.path.basename(video_output_path))
        new_name = '{}_blurred{}'.format(video_name, ext)
        video_output_path = os.path.join(os.path.dirname(video_output_path), new_name)
        if os.path.exists(video_output_path):
            raise ValueError('Give another video output name because the first alternative {} is taken!'.format(
                video_output_path))
        fourcc = cv2.VideoWriter_fourcc(*'MP43')
    fps = camera.get(cv2.CAP_PROP_FPS)
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    # video_output_path.replace('mp4', 'avi')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))
    return video_writer


def main():
    # Read video
    path = '/home/gorfanidis/Datasets/web-images/blur_faces/Jet Blue Flight 292 - SNL_2s.mp4'
    video = cv2.VideoCapture(path)
    video_writer = init_video_recorder(path, video)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    frame_blurred = frame.copy()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    cv2.putText(frame, "Press r to select ROI(s) or press s to skip", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 255), 2)

    # Select boxes
    bboxes = []
    colors = []

    cv2.namedWindow('Tracking')
    cv2.moveWindow('Tracking', 100, 30)
    cv2.imshow("Tracking", frame)

    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    while True:
        while True:
            k = cv2.waitKey(0) & 0xFF  # 113: q, 114: r, 115: s
            if k == 115 or k == 114:
                break
        print(k)
        if k == 115:
            break
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('Tracking', frame)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        print('{} bbox selected: {}'.format(len(bboxes), bbox))
        k = cv2.waitKey(0) & 0xFF
        print(k)
        if k == 113:  # q is pressed
            break

    if bboxes:
        # Initialize tracker with first frame and bounding box
        if all(bboxes[-1]) == 0:
            del bboxes[-1]

        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()

        # Initialize MultiTracker
        for bbox in bboxes:
            multiTracker.add(createTrackerByName('KCF'), frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        frame_blurred = frame.copy()

        if bboxes:

            # Update tracker
            ok, bbox = multiTracker.update(frame)
            if not ok:
                cv2.putText(frame, "Please select a ROI", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                bboxes = []
                while True:
                    # draw bounding boxes over objects
                    # selectROI's default behaviour is to draw box starting from the center
                    # when fromCenter is set to false, you can draw box starting from top left corner
                    bbox = cv2.selectROI('Tracking', frame)
                    bboxes.append(bbox)
                    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                    print("Press q to quit selecting boxes and start tracking")
                    print("Press any other key to select next object")
                    print('{} bbox selected: {}'.format(len(bboxes), bbox))
                    k = cv2.waitKey(0) & 0xFF
                    if k == 113:  # q is pressed
                        break
                while True:
                    if bboxes[-1][2] == 0 and bboxes[-1][3] == 0:
                        del bboxes[-1]

                multiTracker = cv2.MultiTracker_create()
                for bbox in bboxes:
                    multiTracker.add(createTrackerByName('KCF'), frame, bbox)

            for i in range(len(bboxes)):
                x = int(bboxes[i][0])
                y = int(bboxes[i][1])
                w = int(bboxes[i][2])
                h = int(bboxes[i][3])

                sub_face = frame[y:y + h, x:x + w]

                # apply a gaussian blur on this new recangle image
                sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)

                # merge this blurry rectangle to our final image
                frame_blurred[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

            # Display result
            cv2.imshow("Tracking", frame_blurred)
            video_writer.write(frame_blurred)

        else:
            frame_blurred = frame.copy()
            video_writer.write(frame_blurred)
            cv2.putText(frame_blurred, "Press r to select ROI(s) or press Esc to quit", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow("Tracking", frame_blurred)


        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == 114:
            cv2.putText(frame, "Please select a ROI", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            bboxes = []
            while True:
                # draw bounding boxes over objects
                # selectROI's default behaviour is to draw box starting from the center
                # when fromCenter is set to false, you can draw box starting from top left corner
                bbox = cv2.selectROI('Tracking', frame)
                bboxes.append(bbox)
                colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                print('{} bbox selected: {}'.format(len(bboxes), bbox))
                k = cv2.waitKey(0) & 0xFF
                if k == 113:  # q is pressed
                    break
                # print(k)
            if all(bboxes[-1]) == 0:
                del bboxes[-1]

            multiTracker = cv2.MultiTracker_create()
            for bbox in bboxes:
                multiTracker.add(createTrackerByName('KCF'), frame, bbox)


if __name__ == '__main__':
    main()
