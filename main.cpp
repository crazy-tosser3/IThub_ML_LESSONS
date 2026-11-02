#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file>" << std::endl;
        return 1;
    }

    cv::Mat watermark = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    if (watermark.empty()) {
        std::cerr << "Cannot load watermark image" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    int x = 0, y = 0;
    int dx = 5, dy = 5;

    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Empty frame" << std::endl;
            break;
        }

        if (watermark.cols > frame.cols ||
            watermark.rows > frame.rows) {
            cv::resize(watermark, watermark,
                cv::Size(frame.cols / 4, frame.rows / 4));
        }

        x += dx;
        y += dy;
        
        if (x <= 0 || x + watermark.cols >= frame.cols) {
            dx = -dx;
            x = std::max(0, std::min(x, frame.cols - watermark.cols));
        }

        if (y <= 0 || y + watermark.rows >= frame.rows) {
            dy = -dy;
            y = std::max(0, std::min(y, frame.rows - watermark.rows));
        }

        cv::Rect roi(x, y, watermark.cols, watermark.rows);
        cv::Mat imageROI = frame(roi);

        if (watermark.channels() == 4) {
            for (int row = 0; row < watermark.rows; ++row) {
                for (int col = 0; col < watermark.cols; ++col) {
                    cv::Vec4b pixel = watermark.at<cv::Vec4b>(row, col);
                    cv::Vec3b& background = imageROI.at<cv::Vec3b>(row, col);

                    double alpha = pixel[3] / 255.0;

                    for (int c = 0; c < 3; ++c) {
                        background[c] = cv::saturate_cast<uchar>(
                            alpha * pixel[c] +
                            (1.0 - alpha) * background[c]
                        );
                    }
                }
            }
        } else {
            cv::addWeighted(imageROI, 0.5, watermark, 0.5, 0, imageROI);
        }


        std::time_t now = std::time(nullptr);
        std::tm* localTime = std::localtime(&now);

        char timeText[16];
        std::strftime(timeText, sizeof(timeText), "%H:%M:%S", localTime);

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;

        int textX = 20;
        int textY = 40;

        cv::putText(
            frame,
            timeText,
            cv::Point(textX + 2, textY + 2),
            fontFace,
            fontScale,
            cv::Scalar(0, 0, 0),
            thickness + 3,
            cv::LINE_AA
        );

        cv::putText(
            frame,
            timeText,
            cv::Point(textX, textY),
            fontFace,
            fontScale,
            cv::Scalar(255, 255, 255),
            thickness,
            cv::LINE_AA
        );

        cv::imshow("TaskONE", frame);

        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}