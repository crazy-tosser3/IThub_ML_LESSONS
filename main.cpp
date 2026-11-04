#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


    const std::string detection_path = "../face_detection_yunet_2026may.onnx";
    const std::string emotion_model_path = "../emotion_model_duhn.onnx";

    const std::vector<std::string> EMOTIONS =
        {"anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"};

    cv::dnn::Net emotionNet = cv::dnn::readNetFromONNX(emotion_model_path);
    emotionNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    emotionNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "Could not read frame\n";
        return -1;
    }

    try {
        cv::Mat faces;
        int frameNumber = 0;

        std::vector<int> emoLabels;
        std::vector<double> emoConfs;

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

                emoLabels.assign(faces.rows, -1);
                emoConfs.assign(faces.rows, 0.0);

                for (int i = 0; i < faces.rows; ++i) {
                    const float* f = faces.ptr<float>(i);

                    cv::Rect roi(
                        cvRound(f[0]), cvRound(f[1]),
                        cvRound(f[2]), cvRound(f[3])
                    );
                    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
                    if (roi.width < 8 || roi.height < 8) continue;

                    cv::Mat face;
                    cv::resize(frame(roi), face, cv::Size(192, 108));
                    face.convertTo(face, CV_32FC3);
                    cv::cvtColor(face, face, cv::COLOR_BGR2RGB);

                    int dims[] = {1, face.rows, face.cols, 3};
                    cv::Mat blob(4, dims, CV_32F, face.data);  // заголовок, без копии

                    emotionNet.setInput(blob);
                    cv::Mat probs = emotionNet.forward().reshape(1, 1);  // 1x8

                    cv::Point cls;
                    double conf;
                    cv::minMaxLoc(probs, nullptr, &conf, nullptr, &cls);

                    if (cls.x >= 0 && cls.x < static_cast<int>(EMOTIONS.size())) {
                        emoLabels[i] = cls.x;
                        emoConfs[i] = conf;
                    }
                }
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

                if (emoLabels[i] >= 0) {
                    cv::putText(
                        frame,
                        cv::format("%s %.0f%%", EMOTIONS[emoLabels[i]].c_str(), emoConfs[i] * 100),
                        cv::Point(faceBox.x, faceBox.y + faceBox.height + 20),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 255),
                        2
                    );
                }
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