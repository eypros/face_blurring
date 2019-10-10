# Face blurring software v0.2

## What has this project to offer

This is an attempt to create a functional library of tools for blurring faces in videos.

Two different module are supported:
* Fully manual face (or not only face) blurring
* Fully automatic face blurring

The manual module supports:
* Multiple manual annotation of regions to be blurred
* 1, 5, 10 and 100 frame skipping option (without
blurring taken place for those frames that is)
* Skip for an undefined number of frames (waiting for a key stroke or
the end of file)
* Basic key functionality for the above actions

The automatic module supports:
* A parameter to choose which bboxes to consider
* A parameter which controls the size of the blurring area
* A fully automatic facial blurring

## Requirements

* Manual mode

Mainly *openCV*.

* Automatic mode

*OpenCV*, *numpy* and *tensorflow*.

## How to use

### Manual face blurring

Not technically limited to facial images as the ROI annotation is fully manual.

The module *manual_face_blurring_v3.py* uses opencv
for bounding box definition and KFC opencv tracker for bounding box propagation.

Invoking the module is simple by using:

*`python3 manual_face_blurring_v3.py "path/to/video" [--verbose]`*

where verbose is the only optional parameter after the positional
parameter for the video location

It is a fully functional module with basic key functionality. Key being used are:

| Key pressed        | Action performed            |
| ------------- |:-------------:|
| <kbd>r</kbd>      | Define the ROI for the regions to be blurred. Use <kbd>Enter</kbd> after each ROI, twice to add another and once followed by <kbd>q</kbd> to end the ROI annotation|
| <kbd>s</kbd>      | Parses (skips) all remaining frames without applying any blurring. It will be terminated by end of video or another key stroke |
| <kbd>1</kbd>      | Skips 1 frame only and stops to wait for a new keystroke (used for fine video search) |
| <kbd>5</kbd>      | Skips 5 frames and returns to wait for key input |
| <kbd>t</kbd>      | The same for 10 frames skipped |
| <kbd>h</kbd>      | Skips 100 frames and waits for key input |
| <kbd>Esc</kbd>    | Stops the video processing (any video processed up to this moment remains. |

Remarks:
* For key input only English layout is supported (other locale causes the
program to crack).
* Also ROI definition is a bit awkward since it needs 2 times
<kbd>Enter</kbd> key stroke for adding another ROI but only 1 time
<kbd>Enter</kbd> key stroke followed by <kbd>q</kbd> key stroke to
terminate the ROI annotation.

* For v0.4 there has been added the option to press 3 times (3x) the <kbd>Enter</kbd> to exit the ROI annotation mode and enter tracking mode with the annotated bboxes.
* This module creates a new video file named _old_file_**_blurred**_.old_ext_ and it will complain if a file with that name already exists in folder.

### Automatic face blurring

This module uses a tensorflow object detection model from this
[repository](https://github.com/yeephycho/tensorflow-face-detection)
which has trained a mobilenet SSD(single shot multibox detector)
trained on WIDERFACE dataset. This is quite small and fast face detector
and seems to be quite effective also.

Then a simple gaussian filter is applied to all bboxes with confidence
score above a threshold (by default 0.35) and write the frames to a
video file.

it can be invoked by using:
`python3 auto_face_blurring_v2.py "path/to/video"`

Optional arguments are:
* `--score-threshold` with default value `--score-threshold=0.35` which
controls the number of bounging boxes to be blurred. All bboxes with
confidence score below this threshold are ignored.
* `--enlarge-factor` with default value `--enlarge-factor=0.1` which
controls how much the actual facial bbox will be expanded. Default value
equals to 10% for each bounding box in each direction.

Remarks:

* This module creates a new video file named _old_file_**_blurred_auto**._old_ext_.
