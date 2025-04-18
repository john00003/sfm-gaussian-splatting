#pragma once
#include <string>
#include <functional>

class GUIManager {
public:
    GUIManager();
    ~GUIManager();

    void Init();
    void Render(const std::function<void(const std::string&)>& onRunSfM);
    bool ShouldQuit();
    void Shutdown();

    void RequestQuit();

private:
    bool should_quit_ = false;
    void* window_ = nullptr;
    std::string folder_path_ = "images/";
};

int GetSelectedMatchingMethod();
