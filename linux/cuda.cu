#ifndef HEADLESS
#include "window.h"
#endif
#include <iostream>
#include <fstream>
#include <string>

#include <cmath>
#include <complex>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <opencv2/opencv.hpp>

using namespace std::complex_literals;

__global__ void gpu_init(int width, int height, double _grid_size, int cx,
                         int cy, double a, double kx, double ky,
                         thrust::complex<double> *_wave) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || j < 1 || i >= height - 1 || j >= width - 1) {
        return;
    }

    double dx = (j - cx) * _grid_size;
    double dy = (cy - i) * _grid_size;
    double r2 = dx * dx + dy * dy;
    double mag = sqrt(exp(-a * r2));
    thrust::complex<double> ikx = thrust::complex<double>(0, 1) *
                                  thrust::complex<double>(kx * dx + ky * dy);
    _wave[j + i * width] = mag * exp(ikx);
}

// Optimization: Use Shared Memory for stencil computation
__global__ void step_kernel_shared(
    double dt_as, int height, int width, double h2, int mass,
    thrust::complex<double> *__restrict__ wave,
    thrust::complex<double> *__restrict__ new_wave) {
    double dt = dt_as / 24.1888;

    // Block size is assumed to be 32x32
    // Shared memory needs 2 aprons on each side: (32+2)x(32+2) = 34x34
    __shared__ thrust::complex<double> s_wave[34][34];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    // Load data into shared memory with Halo regions
    // Map (tx, ty) [0..31] to shared memory indices [1..32]
    int sx = tx + 1;
    int sy = ty + 1;

    // Load center
    if (gx < width && gy < height) {
        s_wave[sy][sx] = wave[gx + gy * width];
    } else {
        s_wave[sy][sx] = thrust::complex<double>(0, 0);
    }

    // Load Halo: Top / Bottom
    if (ty == 0) {  // Top apron
        if (gy > 0 && gx < width)
            s_wave[0][sx] = wave[gx + (gy - 1) * width];
        else
            s_wave[0][sx] = thrust::complex<double>(0, 0);
    }
    if (ty == blockDim.y - 1) {  // Bottom apron
        if (gy < height - 1 && gx < width)
            s_wave[sy + 1][sx] = wave[gx + (gy + 1) * width];
        else
            s_wave[sy + 1][sx] = thrust::complex<double>(0, 0);
    }

    // Load Halo: Left / Right
    if (tx == 0) {  // Left apron
        if (gx > 0 && gy < height)
            s_wave[sy][0] = wave[(gx - 1) + gy * width];
        else
            s_wave[sy][0] = thrust::complex<double>(0, 0);
    }
    if (tx == blockDim.x - 1) {  // Right apron
        if (gx < width - 1 && gy < height)
            s_wave[sy][sx + 1] = wave[(gx + 1) + gy * width];
        else
            s_wave[sy][sx + 1] = thrust::complex<double>(0, 0);
    }

    __syncthreads();

    if (gx >= 1 && gx < width - 1 && gy >= 1 && gy < height - 1) {
        thrust::complex<double> val = s_wave[sy][sx];
        thrust::complex<double> laplacian = -4.0 * val;
        laplacian += s_wave[sy][sx - 1];  // Left
        laplacian += s_wave[sy][sx + 1];  // Right
        laplacian +=
            s_wave[sy - 1]
                  [sx];  // Top (y is inverted generally but symmetric stencil)
        laplacian += s_wave[sy + 1][sx];  // Bottom

        laplacian /= h2;

        thrust::complex<double> i_unit(0.0, 1.0);
        new_wave[gx + gy * width] =
            val + i_unit * (laplacian / (2 * mass)) * dt;
    }
}

__device__ int device_rgba(double r, double g, double b, double a) {
    unsigned char ru = r * 0xff;
    unsigned char gu = g * 0xff;
    unsigned char bu = b * 0xff;
    unsigned char au = a * 0xff;
    return (au << 24) | (ru << 16) | (gu << 8) | bu;
}

__global__ void complex_to_rgba_kernel(thrust::complex<double> *wave,
                                       int *pixels, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        int idx = j + i * width;
        thrust::complex<double> z = wave[idx];

        double a = thrust::abs(z);
        double y = atan(a) / M_PI;
        z *= 0.5 * y / a;
        double u = z.real();
        double v = z.imag();
        double r = y + 1.5748 * v;
        double g = y - 0.1873 * u - 0.4681 * v;
        double b = y + 1.8556 * u;

        pixels[idx] = device_rgba(r, g, b, 1);
    }
}

class GPUSimulation {
   private:
    int _width, _height;
    double _grid_size, _mass;
    int *_gpu_pixels;
    thrust::complex<double> *_wave, *_new_wave;
    dim3 block, grid;

    void _normalize();

   public:
    GPUSimulation(int width, int height, int cx, int cy, double y_range_ang,
                  double mass_me, double a, double kx, double ky) {
        _width = width;
        _height = height;
        _grid_size = y_range_ang / 0.529177 / height;
        _mass = mass_me;

        block = dim3(32, 32);  // Increased block size for better occupancy
        grid = dim3((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);  // Corrected grid dims

        cudaMalloc(&_wave, width * height * sizeof(thrust::complex<double>));
        cudaMalloc(&_new_wave,
                   width * height * sizeof(thrust::complex<double>));
        cudaMalloc(&_gpu_pixels, width * height * sizeof(int));
        cudaMemset(_wave, 0, width * height * sizeof(thrust::complex<double>));
        cudaMemset(_new_wave, 0,
                   width * height * sizeof(thrust::complex<double>));
        cudaMemset(_gpu_pixels, 0, width * height * sizeof(int));

        gpu_init<<<grid, block>>>(width, height, _grid_size, cx, cy, a, kx, ky,
                                  _wave);
        cudaDeviceSynchronize();
        _normalize();
    }

    void step(double dt) {
        double h2 = _grid_size * _grid_size;

        step_kernel_shared<<<grid, block>>>(dt, _height, _width, h2, _mass,
                                            _wave, _new_wave);

        // _wave.swap(_new_wave);
        thrust::complex<double> *temp = _wave;
        _wave = _new_wave;
        _new_wave = temp;

        _normalize();
    }

    // Helper to copy wave to host for batch output
    void get_wave(std::vector<thrust::complex<double>> &host_wave) {
        cudaMemcpy(host_wave.data(), _wave,
                   _width * _height * sizeof(thrust::complex<double>),
                   cudaMemcpyDeviceToHost);
    }

    int *get_pixel_buffer() {
        complex_to_rgba_kernel<<<grid, block>>>(_wave, _gpu_pixels, _width,
                                                _height);
        cudaDeviceSynchronize();

        return _gpu_pixels;
    }

    ~GPUSimulation() {
        cudaFree(_wave);
        cudaFree(_new_wave);
    }
};

__global__ void scale_wave(thrust::complex<double> *wave, double factor,
                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        wave[idx] /= factor;
    }
}

struct norm_calculator {
    __device__ double operator()(const thrust::complex<double> &z) const {
        return thrust::norm(z);
    }
};

void GPUSimulation::_normalize() {
    int N = _width * _height;

    thrust::device_ptr<thrust::complex<double>> wave_ptr(_wave);

    double integral = thrust::transform_reduce(
        wave_ptr, wave_ptr + N, norm_calculator(), 0.0, thrust::plus<double>());

    integral /= N;
    double factor = std::sqrt(integral);

    int block_tmp = 64;
    int grid_tmp = (N + block_tmp - 1) / block_tmp;
    scale_wave<<<grid_tmp, block_tmp>>>(_wave, factor, N);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    constexpr int sim_w = 256;
    constexpr int sim_h = 256;
    GPUSimulation sim(sim_w, sim_h, sim_w / 4, sim_h / 2, 100, 1, 0.01, 1, 0);

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

            std::cout << "Running CUDA batch mode: " << num_frames
                      << " frames, " << steps_per_frame << " steps/frame" << std::endl;

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < steps_per_frame; ++j) {
                    sim.step(0.01);
                }
                int *gpu_pixels = sim.get_pixel_buffer();
                std::vector<int> host_pixels(sim_w * sim_h);
                cudaMemcpy(host_pixels.data(), gpu_pixels, sim_w * sim_h * sizeof(int), cudaMemcpyDeviceToHost);
                cv::Mat mat(sim_h, sim_w, CV_8UC4, host_pixels.data());
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

        int *gpu_pixels = sim.get_pixel_buffer();
        wave_plot.update([&](int *texture_pixels) {
            cudaMemcpy(texture_pixels, gpu_pixels, sim_w * sim_h * sizeof(int),
                       cudaMemcpyDeviceToHost);
        });
        window.update();
    }
#else
    std::cout << "Compiled in HEADLESS mode. GUI not available." << std::endl;
    std::cout << "Use --batch to run simulation." << std::endl;
#endif
    return 0;
}
