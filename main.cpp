
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

int main() {

    cv::VideoCapture cap(1);

    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true) {

        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Ошибка загрузки изображения!" << std::endl;
            break;
        }

        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); // READY

        cv::Mat mask;

        cv::inRange(
            hsv,
            cv::Scalar(0, 10, 0),
            cv::Scalar(179, 255, 255),
            mask
        );

        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5)
        );

        cv::morphologyEx(
            mask,
            mask,
            cv::MORPH_CLOSE,
            kernel,
            cv::Point(-1, -1),
            2
        );

        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(
            mask.clone(),
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        cv::Mat result = frame.clone();

        int rectangle_count = 0;

        double minArea = frame.rows * frame.cols * 0.005;
        double maxArea = frame.rows * frame.cols * 0.85;

        for (const auto& contour : contours) {

            double area = cv::contourArea(contour);

            if (area < minArea || area > maxArea)
                continue;

            cv::RotatedRect rect = cv::minAreaRect(contour);

            double width = rect.size.width;
            double height = rect.size.height;

            if (width <= 0 || height <= 0)
                continue;

            double rectArea = width * height;

            double rectangularity = area / rectArea;

            if (rectangularity < 0.65)
                continue;

            double aspectRatio =
                std::max(width, height) /
                std::min(width, height);

            if (aspectRatio > 8.0)
                continue;

            cv::Point2f points[4];
            rect.points(points);

            std::vector<cv::Point> box;

            for (int i = 0; i < 4; i++) {
                box.push_back(points[i]);
            }

            cv::polylines(
                result,
                box,
                true,
                cv::Scalar(0, 255, 0),
                3
            );

            cv::Point center(
                cvRound(rect.center.x),
                cvRound(rect.center.y)
            );

            cv::Point2f edge1 = points[1] - points[0];
            cv::Point2f edge2 = points[2] - points[1];

            cv::Point2f usedEdge =
                cv::norm(edge1) >= cv::norm(edge2)
                ? edge1
                : edge2;

            double angle = std::atan2(
                usedEdge.y,
                usedEdge.x
            ) * 180.0 / CV_PI;

            while (angle >= 90)
                angle -= 180;

            while (angle < -90)
                angle += 180;

            cv::circle(
                result,
                center,
                5,
                cv::Scalar(255, 0, 0),
                -1
            );

            rectangle_count++;

            cv::putText(
                result,
                "Rect #" + std::to_string(rectangle_count),
                cv::Point(center.x + 10, center.y - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 255),
                2
            );

            cv::putText(
                result,
                std::to_string(static_cast<int>(std::round(angle))) + " deg",
                cv::Point(center.x + 10, center.y + 20),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(255, 0, 0),
                2
            );

            std::cout
                << "Rect #" << rectangle_count
                << " | Center: " << center
                << " | Angle: " << angle
                << std::endl;
        }

        std::cout
            << "\nНайдено прямоугольников: "
            << rectangle_count
            << std::endl;

        cv::imshow("Original", frame);
        cv::imshow("HSV Mask", mask);
        cv::imshow("Detected Rectangles", result);

        char key = static_cast<char>(cv::waitKey(30));

        if (key == 'q' || key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
