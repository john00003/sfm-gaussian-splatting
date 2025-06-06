#include "gui.h"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <cstring>

static int matching_method = 0; // 0: BF, 1: Sequential, 2: Window + Anchor 
int GetSelectedMatchingMethod() {
    return matching_method;
}

GUIManager::GUIManager() {}
GUIManager::~GUIManager() {}

void GUIManager::Init() {
    if (!glfwInit()) {
        std::cerr << "[ERROR] Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(800, 600, "SfM GUI", NULL, NULL);
    if (!window) {
        std::cerr << "[ERROR] Failed to create GLFW window" << std::endl;
        return;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    window_ = (void*)window;
}

void GUIManager::Render(const std::function<void(const std::string&)>& onRunSfM) {
    GLFWwindow* window = (GLFWwindow*)window_;
    glfwPollEvents();
    if (glfwWindowShouldClose(window)) {
        should_quit_ = true;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Incremental SfM");

    ImGui::Text("A group project by John Tyler and Bob Bu");
    ImGui::Text("ALBERTA CANADA, APRIL 2025");
    ImGui::Separator();
    ImGui::Text("jetyler@ualberta.ca | sbu1@ualberta.ca");
    ImGui::Text("Department of Computing Science, University of Alberta");
    ImGui::Separator();

    ImGui::Text("Input image folder:");
    static char buf[512];
    std::strncpy(buf, folder_path_.c_str(), sizeof(buf));
    if (ImGui::InputText("##Folder", buf, sizeof(buf))) {
        folder_path_ = std::string(buf);
    }

    ImGui::Text("Matching Strategy:");
    ImGui::RadioButton("Brute Force (BF)", &matching_method, 0);
    ImGui::RadioButton("Sequential", &matching_method, 1);
    ImGui::RadioButton("Window + Anchor", &matching_method, 2);

    if (ImGui::Button("Run SfM")) {
        const char* method_str = nullptr;
        switch (matching_method) {
            case 0: method_str = "Brute Force (BF)"; break;
            case 1: method_str = "Sequential"; break;
            case 2: method_str = "Window + Anchor"; break;
        }

        std::cout << "[INFO] Selected Matching Strategy: " << method_str << std::endl;
        onRunSfM(folder_path_);
    }

    ImGui::SameLine();
    if (ImGui::Button("Quit")) {
        should_quit_ = true;
    }

    ImGui::End();
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
}

bool GUIManager::ShouldQuit() {
    return should_quit_;
}

void GUIManager::RequestQuit() {
    should_quit_ = true;
}

void GUIManager::Shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    GLFWwindow* window = (GLFWwindow*)window_;
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
        window_ = nullptr;
    }
}
