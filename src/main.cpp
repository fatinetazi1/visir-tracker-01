#include "types.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay( Mat frame );

// Global variables
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

// Runner
int main(int argc, const char** argv) {
    
    if( argc != 3 ) {
        cout << "Usage: ./visir-tracker <path_to_haarcascade_frontalface_alt.xml>                                 <path_to_haarcascade_eye_tree_eyeglasses.xml>" << endl;
        return 1;
    }
    
    VideoCapture camera; // Camera
    Mat img; // Image
    time_t start, end; // Start and end times
    int frames = 0; // Number of frames
    double seconds = 0; // Seconds elapsed
    int fps = 0; // Frames per second
    
    // Problem 1.1
    cout << "Problem 1.1" << endl;
    time(&start); // Start time
    
	if (!camera.open(0)) {
		printf("Can't find a camera\n");
		return 1;
	};
	
	// Loop 1
	for(;;) {
        frames++;
		camera >> img;
		imshow("Camera", img);
        
        time(&end); // End Time
        seconds = difftime (end, start); // Time elapsed
        if (seconds >= 2) {
            fps  = frames / seconds; // Frames per second
            cout << "Frames  : " << frames << endl;
            cout << "Seconds  : " << seconds << endl;
            cout << "Frames per second : " << fps << endl << endl;
            frames = 0;
            time(&start); // New start time
        }
        
		int key = waitKey(5);
		if (key == 27 || key == 'q') break;
	}
    
	camera.release();
    
    // Problem 1.2
    cout << "Problem 1.2" << endl;
    
    // Loading the cascades
    if(!face_cascade.load(argv[1]) ) {
        cout << "Error loading face cascade" << endl;
        return 1;
    };
    
    if(!eyes_cascade.load(argv[2])) {
        cout << "Error loading eyes cascade" << endl;
        return 1;
    };
    
    frames = 0;
    time(&start); // Start time
    
    // Reading the video stream
    if (!camera.open(0)) {
        cout << "Error opening video camera" << endl;
        return 1;
    }
    
    // Loop 2
    for(;;) {
        frames++;
        camera >> img;
        imshow("Camera", img);
        detectAndDisplay(img); // Applying the classifier to the frame
        int key = waitKey(5);
        if (key == 27 || key == 'q') break;
    }
    
    time(&end); // End Time
    seconds = difftime (end, start); // Time elapsed
    
    fps  = frames / seconds; // Frames per second
    cout << "Frames  : " << frames << endl;
    cout << "Seconds  : " << seconds << endl;
    cout << "Frames per second : " << fps << endl;
    
    camera.release();
    
	return 0;
}

// Face and eye detection
void detectAndDisplay(Mat frame) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY); // Convert to grayscale
    equalizeHist(frame_gray, frame_gray); // Applying histogram equalization
    
    // Face Detection
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    
    for (size_t i = 0; i < faces.size(); i++) {
        Point face_center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, face_center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
        Mat faceROI = frame_gray(faces[i]);
        
        // Detect eyes in face
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        
        for ( size_t j = 0; j < eyes.size(); j++ ) {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
        }
    }
    // Show results
    imshow( "Camera - Face Detection", frame);
}
