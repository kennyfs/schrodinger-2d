#include "window.h"

// class Texture
Texture::Texture(int width, int height) {
    _width = width;
    _height = height;
    _rect = { 0, 0, width, height };
}

Texture::~Texture() {
    SDL_DestroyTexture(_texture);
}

int Texture::width() const { return _width; }
int Texture::height() const { return _height; }

void Texture::move(int x, int y) {
    _rect.x = x;
    _rect.y = y;
}

void Texture::resize(int w, int h) {
    _rect.w = w;
    _rect.h = h;
}

void Texture::update(std::function<void(int*)> callback) {
    int *pixels, pitch;
    SDL_LockTexture(_texture, nullptr, (void**)&pixels, &pitch);
    callback(pixels);
    SDL_UnlockTexture(_texture);
}

// class Window
bool Window::_initialized = false;

Window::Window(int width, int height) {
    _width = width;
    _height = height;
    _closed = false;
    if (!_initialized) {
        SDL_Init(SDL_INIT_VIDEO);
        _initialized = true;
    }
    SDL_CreateWindowAndRenderer(width, height, 0, &_window, &_renderer);
}

Window::~Window() {
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);
}

int Window::width() const { return _width; }
int Window::height() const { return _height; }
bool Window::closed() const { return _closed; }

void Window::update() {
    SDL_PollEvent(&_event);
    if (_event.type == SDL_QUIT) _closed = true;
    SDL_RenderClear(_renderer);
    for (auto &x : _textures)
        SDL_RenderCopy(_renderer, x._texture, nullptr, &x._rect);
    SDL_RenderPresent(_renderer);
}

Texture& Window::create_texture(int width, int height) {
    Texture &texture = _textures.emplace_back(width, height);
    texture._texture = SDL_CreateTexture(
        _renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        width, height
    );
    return texture;
}

