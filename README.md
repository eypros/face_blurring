# Face blurring software v0.1

## What has this project to offer

This is an attempt to create a functional library of tools for blurring faces in videos.

So far, mainly manual bounding box annotation is supported via opencv.

Nevertheless, this module supports:
* Multiple manual annotation of regions to be blurred
* 1, 5, 10 and 100 frame skipping option (without
blurring taken place for those frames that is)
* Skip for an undefined number of frames (waiting for a key stroke or
the end of file)
* Basic key functionality for the above actions

## What's to come in a (hopefully) near future

I am hoping to populate this respository with a fully automatic face blurring module as well as a semi-automatic

## How to use

At the moment the only module working is *manual_face_blurring.py* which uses opencv
for bounding box definition and KFC opencv tracker for bounding box propagation.

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

