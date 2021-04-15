# Video Analysis

### Analysis of a new broadcast.

## Features


- **(a)** Detect shots in the videos. A shot is a set of consecutive frames with a smooth camera
motion.
- **(b)** (Manually) Annotate shot boundaries in the video. How would you evaluate how well
you are detecting the shots? Why is this a good metric? Compute the performance
according to your proposed evaluation metric.
- **(c)** Detect the news company’s logo. Please do not assume that you know the location of
the logo within a frame, it should be found automatically.
- **(d)** Detect faces in the video.
- **(e)** Perform face tracking by correctly associating a face detection in the previous frame to
a face detection in the current frame.
- **(f)** You will be given a dataset of female and male faces. Train a classifier that can predict
whether a face is female or male. For each face track in the news video predict whether
it is female or male. To do this you will take a few images of faces, compute image
features and train a male-vs-female classifier, e.g., SVM, NN. Once trained, you will
predict the gender of each face detection in the video. That is, you’ll take each face
detection (a crop in the image specified by the face box), compute appropriate image
features, and use your classifier to predict the gender of the face. How would you decide
whether a full face track is female or male?
- **(g)** Visualize your results: produce a video in which you show a bounding box around
the detected company logo, and bounding boxes around the detected faces. Each face
bounding box should have text indicated whether the face is male or female.
