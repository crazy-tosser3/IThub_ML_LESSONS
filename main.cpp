#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

namespace fs = std::filesystem;

struct Config {
    std::string input_path;
    std::string out_path;
    std::string model_path = "model/model.onnx";
    std::string device = "auto"; // auto, cuda, metal, cpu
};

void print_usage(const char* prog) {
    std::cout << "Использование:\n  " << prog
              << " --input <path_to_photo_or_video> [--out <save_path>] [--model <model.onnx>] [--device <auto|cuda|metal|cpu>]\n\n"
              << "Управление в интерактивном режиме (без --out):\n"
              << "  R     - повернуть кадр на 90 градусов по часовой стрелке\n"
              << "  C     - включить / выключить раскрашивание через ONNX модель\n"
              << "  Space - пауза (для видео)\n"
              << "  Q/ESC - выход\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            cfg.input_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            cfg.out_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            cfg.device = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            return false;
        }
    }
    return !cfg.input_path.empty();
}

std::string resolve_model_path(const std::string& requested) {
    if (fs::exists(requested)) return requested;
    if (fs::exists("../" + requested)) return "../" + requested;
    if (fs::exists("model.onnx")) return "model.onnx";
    return requested;
}

class Colorizer {
public:
    Colorizer(const std::string& model_path, const std::string& device)
        : env_(ORT_LOGGING_LEVEL_WARNING, "Razukrashka2") {

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        setup_execution_provider(session_options, device);

#ifdef _WIN32
        std::wstring wpath(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        auto in_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = in_name.get();

        auto out_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = out_name.get();

        auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() == 4) {
            if (in_shape[3] == 1 || in_shape[3] == 3) {
                is_nhwc_ = true;
                img_h_ = (in_shape[1] > 0) ? static_cast<int>(in_shape[1]) : 256;
                img_w_ = (in_shape[2] > 0) ? static_cast<int>(in_shape[2]) : 256;
            } else {
                is_nhwc_ = false;
                img_h_ = (in_shape[2] > 0) ? static_cast<int>(in_shape[2]) : 256;
                img_w_ = (in_shape[3] > 0) ? static_cast<int>(in_shape[3]) : 256;
            }
        }
        std::cout << "[ONNX] Модель загружена: " << model_path
                  << " (" << img_w_ << "x" << img_h_ << ", " << (is_nhwc_ ? "NHWC" : "NCHW") << ")\n";
    }

    cv::Mat colorize_hd(const cv::Mat& bgr_frame) {
        if (bgr_frame.empty()) return bgr_frame;

        const int orig_h = bgr_frame.rows;
        const int orig_w = bgr_frame.cols;

        cv::Mat rgb_float;
        cv::cvtColor(bgr_frame, rgb_float, cv::COLOR_BGR2RGB);
        rgb_float.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

        cv::Mat gray_full;
        cv::cvtColor(rgb_float, gray_full, cv::COLOR_RGB2GRAY);

        cv::Mat gray_small;
        cv::resize(gray_full, gray_small, cv::Size(img_w_, img_h_), 0, 0, cv::INTER_LINEAR);
        if (!gray_small.isContinuous()) {
            gray_small = gray_small.clone();
        }

        std::vector<int64_t> input_shape = is_nhwc_
            ? std::vector<int64_t>{1, img_h_, img_w_, 1}
            : std::vector<int64_t>{1, 1, img_h_, img_w_};

        size_t input_numel = static_cast<size_t>(img_h_ * img_w_);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, gray_small.ptr<float>(), input_numel,
            input_shape.data(), input_shape.size()
        );

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );

        float* out_data = output_tensors.front().GetTensorMutableData<float>();

        cv::Mat pred_rgb_small(img_h_, img_w_, CV_32FC3);
        if (is_nhwc_) {
            std::memcpy(pred_rgb_small.data, out_data, input_numel * 3 * sizeof(float));
        } else {
            std::vector<cv::Mat> ch(3);
            for (int c = 0; c < 3; ++c) {
                ch[c] = cv::Mat(img_h_, img_w_, CV_32FC1, out_data + c * input_numel);
            }
            cv::merge(ch, pred_rgb_small);
        }

        cv::Mat pred_rgb_full;
        cv::resize(pred_rgb_small, pred_rgb_full, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_CUBIC);
        cv::threshold(pred_rgb_full, pred_rgb_full, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(pred_rgb_full, pred_rgb_full, 0.0, 0.0, cv::THRESH_TOZERO);

        cv::Mat pred_ycrcb;
        cv::cvtColor(pred_rgb_full, pred_ycrcb, cv::COLOR_RGB2YCrCb);

        std::vector<cv::Mat> ycrcb_channels(3);
        cv::split(pred_ycrcb, ycrcb_channels);
        ycrcb_channels[0] = gray_full; // Подставляем оригинальную резкую яркость!
        cv::merge(ycrcb_channels, pred_ycrcb);

        cv::Mat sharp_bgr_float, sharp_bgr_u8;
        cv::cvtColor(pred_ycrcb, sharp_bgr_float, cv::COLOR_YCrCb2BGR);
        sharp_bgr_float.convertTo(sharp_bgr_u8, CV_8UC3, 255.0);

        return sharp_bgr_u8;
    }

private:
    void setup_execution_provider(Ort::SessionOptions& options, const std::string& device) {
        auto providers = Ort::GetAvailableProviders();
        auto has_provider = [&](const std::string& name) {
            return std::find(providers.begin(), providers.end(), name) != providers.end();
        };

        std::cout << "[ONNX] Доступные провайдеры: ";
        for (const auto& p : providers) std::cout << p << " ";
        std::cout << "\n";

        bool want_cuda = (device == "auto" || device == "cuda");
        bool want_metal = (device == "auto" || device == "metal" || device == "coreml");

        if (want_cuda && has_provider("CUDAExecutionProvider")) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[ONNX] Активирован ускоритель: CUDAExecutionProvider\n";
                return;
            } catch (const Ort::Exception& e) {
                std::cerr << "[ONNX] Не удалось включить CUDA: " << e.what() << "\n";
            }
        }

#if defined(__APPLE__)
        if (want_metal && has_provider("CoreMLExecutionProvider")) {
            try {
                uint32_t coreml_flags = 0;
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags));
                std::cout << "[ONNX] Активирован ускоритель: CoreMLExecutionProvider (Apple Metal / ANE)\n";
                return;
            } catch (const Ort::Exception& e) {
                std::cerr << "[ONNX] Не удалось включить Metal/CoreML: " << e.what() << "\n";
            }
        }
#endif
        std::cout << "[ONNX] Используется CPUExecutionProvider\n";
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int img_h_ = 256;
    int img_w_ = 256;
    bool is_nhwc_ = true;
};

bool is_image_file(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    const std::vector<std::string> img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"};
    return std::find(img_exts.begin(), img_exts.end(), ext) != img_exts.end();
}

cv::Mat apply_transform(const cv::Mat& frame, int rotation_steps, bool do_colorize, Colorizer& colorizer) {
    cv::Mat result = frame.clone();
    int rot = ((rotation_steps % 4) + 4) % 4;
    if (rot == 1) {
        cv::rotate(result, result, cv::ROTATE_90_CLOCKWISE);
    } else if (rot == 2) {
        cv::rotate(result, result, cv::ROTATE_180);
    } else if (rot == 3) {
        cv::rotate(result, result, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    if (do_colorize) {
        result = colorizer.colorize_hd(result);
    }
    return result;
}

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    cfg.model_path = resolve_model_path(cfg.model_path);
    if (!fs::exists(cfg.model_path)) {
        std::cerr << "Ошибка: файл модели не найден по пути: " << cfg.model_path << "\n";
        return 1;
    }

    if (!fs::exists(cfg.input_path)) {
        std::cerr << "Ошибка: входной файл не найден: " << cfg.input_path << "\n";
        return 1;
    }

    try {
        Colorizer colorizer(cfg.model_path, cfg.device);
        bool is_image = is_image_file(cfg.input_path);

        // РЕЖИМ 1: Прямое сохранение в файл (--out указан)
        if (!cfg.out_path.empty()) {
            if (is_image) {
                cv::Mat img = cv::imread(cfg.input_path, cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Не удалось прочитать изображение: " << cfg.input_path << "\n";
                    return 1;
                }
                cv::Mat out_img = colorizer.colorize_hd(img);
                if (!cv::imwrite(cfg.out_path, out_img)) {
                    std::cerr << "Не удалось сохранить изображение: " << cfg.out_path << "\n";
                    return 1;
                }
                std::cout << "Готово! Изображение сохранено в: " << cfg.out_path << "\n";
            } else {
                cv::VideoCapture cap(cfg.input_path);
                if (!cap.isOpened()) {
                    std::cerr << "Не удалось открыть видео: " << cfg.input_path << "\n";
                    return 1;
                }
                int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                double fps = cap.get(cv::CAP_PROP_FPS);
                if (fps <= 0.0) fps = 30.0;
                int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

                cv::VideoWriter writer(
                    cfg.out_path,
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps,
                    cv::Size(width, height)
                );

                if (!writer.isOpened()) {
                    std::cerr << "Не удалось создать выходной видеофайл: " << cfg.out_path << "\n";
                    return 1;
                }

                cv::Mat frame;
                int idx = 0;
                while (cap.read(frame)) {
                    cv::Mat colorized = colorizer.colorize_hd(frame);
                    writer.write(colorized);
                    ++idx;
                    if (idx % 10 == 0 || idx == total_frames) {
                        std::cout << "\rОбработано кадров: " << idx << " / " << total_frames << std::flush;
                    }
                }
                std::cout << "\nГотово! Видео сохранено в: " << cfg.out_path << "\n";
            }
            return 0;
        }

        // РЕЖИМ 2: Интерактивный просмотр через imshow (--out НЕТ)
        const std::string win_name = "Razukrashka2 [R: Rotate 90 | C: Colorize | Q/ESC: Exit]";
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);

        int rotation_steps = 0;
        bool colorize_enabled = false;

        if (is_image) {
            cv::Mat orig = cv::imread(cfg.input_path, cv::IMREAD_COLOR);
            if (orig.empty()) {
                std::cerr << "Не удалось загрузить картинку: " << cfg.input_path << "\n";
                return 1;
            }

            bool need_update = true;
            cv::Mat display_frame;

            while (true) {
                if (need_update) {
                    display_frame = apply_transform(orig, rotation_steps, colorize_enabled, colorizer);
                    cv::imshow(win_name, display_frame);
                    need_update = false;
                }

                int key = cv::waitKey(30) & 0xFF;
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                } else if (key == 'r' || key == 'R') {
                    rotation_steps = (rotation_steps + 1) % 4;
                    need_update = true;
                } else if (key == 'c' || key == 'C') {
                    colorize_enabled = !colorize_enabled;
                    std::cout << "Колоризация: " << (colorize_enabled ? "ВКЛ" : "ВЫКЛ") << "\n";
                    need_update = true;
                }
            }
        } else {
            cv::VideoCapture cap(cfg.input_path);
            if (!cap.isOpened()) {
                std::cerr << "Не удалось открыть видео: " << cfg.input_path << "\n";
                return 1;
            }

            double fps = cap.get(cv::CAP_PROP_FPS);
            int delay = (fps > 0.0) ? std::max(1, static_cast<int>(1000.0 / fps)) : 30;

            cv::Mat current_frame;
            bool paused = false;

            while (true) {
                if (!paused) {
                    if (!cap.read(current_frame)) {
                        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        if (!cap.read(current_frame)) break;
                    }
                }

                if (!current_frame.empty()) {
                    cv::Mat display_frame = apply_transform(current_frame, rotation_steps, colorize_enabled, colorizer);
                    cv::imshow(win_name, display_frame);
                }

                int key = cv::waitKey(delay) & 0xFF;
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                } else if (key == 'r' || key == 'R') {
                    rotation_steps = (rotation_steps + 1) % 4;
                } else if (key == 'c' || key == 'C') {
                    colorize_enabled = !colorize_enabled;
                    std::cout << "Колоризация: " << (colorize_enabled ? "ВКЛ" : "ВЫКЛ") << "\n";
                } else if (key == ' ') {
                    paused = !paused;
                }
            }
        }

        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << "\n";
        return 1;
    }

    return 0;
}