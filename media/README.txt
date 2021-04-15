The directories clip_* contain 3 short clips from various news programs. train_data contains training examples for male and female faces. If you use these, please cite:

Gary B. Huang, Marwan Mattar, Honglak Lee, and Erik Learned-Miller.
Learning to Align from Scratch.
Advances in Neural Information Processing Systems (NIPS), 2012.

Each male/female face has accompanying mat file. It contains 4 points: left eye, right eye, nose and mouth. Based on these points you should crop out the image such that you get aligned faces. This will make your classifier to work much better. Think what kind of crop would make sense.

Some code for detection is included in the 'code' directory. The directory 'dpm' has the Deformable PartModel detector by felzenswalb et al. Look at function demo.m to see how to run it. You can use the already trained face detector 'face_final.mat' provided in the code directory to perform face detection with this detector. Cite the paper in here: http://markusmathias.bitbucket.org/2014_eccv_face_detection/, if you use this model.
