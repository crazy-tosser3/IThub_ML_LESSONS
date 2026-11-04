#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


    const std::string detection_path =
        "../face_detection_yunet_2026may.onnx";
    const std::string recognition_path =
        "../face_recognition_sface_2021dec_int8.onnx";

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "Could not read frame\n";
        return -1;
    }



    try {
        cv::Mat faces;
        int frameNumber = 0;

        cv::Ptr<cv::FaceDetectorYN> detector =
            cv::FaceDetectorYN::create(
                detection_path, "", frame.size(),
                0.9f, 0.3f, 5000
            );

        while (true) {
            cap >> frame;
            cv::flip(frame, frame, 1);

            if (frame.empty()) {
                std::cerr << "Could not read frame\n";
                break;
            }

            if (frameNumber % 3 == 0) {
                detector->setInputSize(frame.size());
                detector->detect(frame, faces);
            }

            ++frameNumber;

            for (int i = 0; i < faces.rows; ++i) {
                const float* face = faces.ptr<float>(i);

                cv::Rect faceBox(
                    cvRound(face[0]),
                    cvRound(face[1]),
                    cvRound(face[2]),
                    cvRound(face[3])
                );

                cv::rectangle(
                    frame,
                    faceBox,
                    cv::Scalar(0, 255, 0),
                    2
                );

                cv::putText(
                    frame,
                    cv::format("Face: %.2f", face[14]),
                    cv::Point(faceBox.x, std::max(20, faceBox.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            cv::imshow("Face detection", frame);

            if (cv::waitKey(30) == 'q') {
                break;
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << '\n';
        return -1;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}