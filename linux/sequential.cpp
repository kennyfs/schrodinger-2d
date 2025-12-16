#ifndef HEADLESS
#include "window.h"
#endif

#include <cmath>
#include <complex>
#include <vector>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std::complex_literals;

class Simulation {
    int _width, _height;
    double _grid_size, _mass;
    std::vector<std::complex<double>> _wave, _new_wave;

    void _normalize() {
        double integral = 0;
        for (auto &z : _wave) integral += std::norm(z);
        integral /= _wave.size();
        for (auto &z : _wave) z /= std::sqrt(integral);
    }

   public:
    Simulation(int width, int height, int cx, int cy, double y_range_ang,
               double mass_me, double a, double kx, double ky) {
        _width = width;
        _height = height;
        _grid_size = y_range_ang / 0.529177 / height;
        _mass = mass_me;
        _wave.resize(width * height);
        _new_wave.resize(width * height);
        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                double dx = (j - cx) * _grid_size;
                double dy = (cy - i) * _grid_size;
                double r2 = dx * dx + dy * dy;
                double mag = std::sqrt(std::exp(-a * r2));
                auto ikx = 1i * (kx * dx + ky * dy);
                _wave[j + i * width] = mag * std::exp(ikx);
            }
        }
        _normalize();
    }

    const auto &wave() { return _wave; };

    void step(double dt_as) {
        double dt = dt_as / 24.1888;
        double h2 = _grid_size * _grid_size;
        for (int i = 1; i < _height - 1; i++) {
            for (int j = 1; j < _width - 1; j++) {
                int k = j + i * _width;
                auto laplacian = -4.0 * _wave[k];
                laplacian += _wave[k - 1];
                laplacian += _wave[k + 1];
                laplacian += _wave[k - _width];
                laplacian += _wave[k + _width];
                laplacian /= h2;
                _new_wave[k] = _wave[k] + 1i * (laplacian / (2 * _mass)) * dt;
            }
        }
        _wave.swap(_new_wave);
        _normalize();
    }
};

int rgba(double r, double g, double b, double a) {
    unsigned char ru = r * 0xff;
    unsigned char gu = g * 0xff;
    unsigned char bu = b * 0xff;
    unsigned char au = a * 0xff;
    return (bu << 24) | (gu << 16) | (ru << 8) | au;
}

int complex2rgba(std::complex<double> z) {
    double a = std::abs(z);
    double y = std::atan(a) / M_PI;
    if (a < 1e-9 || y < 0.01) {
        return rgba(0.2, 0.2, 0.2, 1);
    }
    z *= 0.5 * y / a;
    double u = std::real(z);
    double v = std::imag(z);
    double r = y + 1.5748 * v;
    double g = y - 0.1873 * u - 0.4681 * v;
    double b = y + 1.8556 * u;
    r = std::max(0.0, std::min(1.0, r));
    g = std::max(0.0, std::min(1.0, g));
    b = std::max(0.0, std::min(1.0, b));
    return rgba(r, g, b, 1);
}

int main(int argc, char *argv[]) {
    constexpr int sim_w = 256;
    constexpr int sim_h = 256;
    Simulation sim(sim_w, sim_h, sim_w / 4, sim_h / 2, 100, 1, 0.01, 1, 0);

    if (argc > 1) {
        if (std::string(argv[1]) == "--batch") {
            if (argc < 6) {
                std::cerr
                    << "Usage: " << argv[0]
                    << " --batch <output_file> <num_frames> <steps_per_frame> <fps>"
                    << std::endl;
                return 1;
            }
            std::string output_file = argv[2];
            int num_frames = std::stoi(argv[3]);
            int steps_per_frame = std::stoi(argv[4]);
            int fps = std::stoi(argv[5]);

            cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('H','2','6','4'), fps, cv::Size(sim_w, sim_h));
            if (!writer.isOpened()) {
                std::cerr << "Failed to open video writer: " << output_file << std::endl;
                return 1;
            }

            std::cout << "Running batch mode: " << num_frames << " frames, "
                      << steps_per_frame << " steps/frame" << std::endl;

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < steps_per_frame; ++j) {
                    sim.step(0.01);
                }
                auto &wave = sim.wave();
                std::vector<int> pixels(sim_w * sim_h);
                for (int k = 0; k < sim_w * sim_h; ++k) {
                    pixels[k] = complex2rgba(wave[k]);
                }
                cv::Mat mat(sim_h, sim_w, CV_8UC4, pixels.data());
                cv::Mat bgr;
                cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
                writer.write(bgr);
                if (i % 10 == 0)
                    std::cout << "Frame " << i << "/" << num_frames << "\r"
                              << std::flush;
            }
            writer.release();
            std::cout << "Batch simulation complete." << std::endl;
            return 0;
        }
    }

#ifndef HEADLESS
    Window window(1280, 960);
    Texture &background = window.create_texture(1, 1);
    background.move(0, 0);
    background.resize(window.width(), window.height());
    background.update([](int *color) { *color = 0x333333ff; });

    constexpr double aspect = (double)sim_w / sim_h;
    Texture &wave_plot = window.create_texture(sim_w, sim_h);
    wave_plot.move(0, 0);
    wave_plot.resize(window.height() * aspect, window.height());

    while (!window.closed()) {
        for (int i = 0; i < 200; i++) sim.step(0.01);
        auto &wave = sim.wave();
        wave_plot.update([&](int *pixels) {
            for (int i = 0; i < sim_w * sim_h; i++)
                pixels[i] = complex2rgba(wave[i]);
        });
        window.update();
    }
#else
    std::cout << "Compiled in HEADLESS mode. GUI not available." << std::endl;
    std::cout << "Use --batch to run simulation." << std::endl;
#endif
    return 0;
}
