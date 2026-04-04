# CS 444/544 Project

Logan Nunno and Quinn Sena

# TODO

- Make the project

## General Idea or Plan

Build a CNN that will be used to detect faces and the person they belong to. Have a way of adding a new user to the
program so it will gather enough info so that it can recognize the person with good enough accuracy and when a new use
is added train the model based on the known users and save the model for later use.
The CNN should be advance enough so it can later be updated or used to detect if there is movement in the video capture
for liveness detection. The way I want to do liveness detection is with either checking for blinking or I have seen
things about ocular flow can be seen. I want to have a general function that will 1 add a user 2 detect who the user is
and give them some stored info. Later I want to use to save passwords for each user or 2fa values.

The basic feature should be the facial recognition and then a second function will check for liveness so if that does not
work it is ok. I also want to make this work in the command line for anyone, so I want a way to package the python
project including everything needed and then produce an exe or something similar. 