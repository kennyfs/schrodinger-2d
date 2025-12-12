#ifndef WINDOW_H
#define WINDOW_H

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <functional>
#include <list>

class Texture {
    SDL_Texture *_texture;
    int _width, _height;
    SDL_Rect _rect;

public:
    Texture(int width, int height);
    ~Texture();
    int width() const;
    int height() const;
    void move(int x, int y);
    void resize(int w, int h);
    void update(std::function<void(int*)> callback);
    friend class Window;
};

class Window {
    static bool _initialized;
    int _width, _height;
    bool _closed;
    SDL_Window *_window;
    SDL_Renderer *_renderer;
    SDL_Event _event;
    std::list<Texture> _textures;

public:
    Window(int width, int height);
    ~Window();
    int width() const;
    int height() const;
    bool closed() const;
    void update();
    Texture& create_texture(int width, int height);
};

#endif // WINDOW_H

