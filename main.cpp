#include <iostream>
#include <opencv2/opencv.hpp>

void addChannel10Percent(cv::Mat& image, int channel) {
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(y, x);
            pixel[channel] = cv::saturate_cast<uchar>(pixel[channel] * 1.1);
        }
    }
}

cv::Mat rotate180(const cv::Mat& image) {
    cv::Mat rotated;
    cv::rotate(image, rotated, cv::ROTATE_180);
    return rotated;
}

int main(int argc, char** argv) {
    std::string imagePath = argc > 1 ? argv[1] : "image.jpg";
    cv::Mat original = cv::imread(imagePath);
    if (original.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение \"" << imagePath << "\"\n"
                  << "Использование: final <путь_к_изображению>\n";
        return 1;
    }

    bool addRed = false;
    bool addGreen = false;
    bool addBlue = false;
    bool flip180 = false;

    std::cout << "Управление:\n"
              << "  R — добавить/убрать 10% к красному каналу\n"
              << "  G — добавить/убрать 10% к зелёному каналу\n"
              << "  B — добавить/убрать 10% к синему каналу\n"
              << "  T — повернуть изображение на 180°\n"
              << "  ESC — выход\n";

    cv::namedWindow("Image");
    while (true) {
        cv::Mat result = original.clone();

        if (addBlue) addChannel10Percent(result, 0);
        if (addGreen) addChannel10Percent(result, 1);
        if (addRed) addChannel10Percent(result, 2);

        if (flip180) result = rotate180(result);

        cv::imshow("Image", result);

        int key = cv::waitKey(0);
        switch (key) {
            case 'r':
            case 'R': addRed = !addRed; break;
            case 'g':
            case 'G': addGreen = !addGreen; break;
            case 'b':
            case 'B': addBlue = !addBlue; break;
            case 't':
            case 'T': flip180 = !flip180; break;
            case 27: return 0;
            default: break;
        }
    }
}
