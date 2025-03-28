CMAKE_BUILD_DIR := build
TARGET_NAME := sfm_ceres

.PHONY: all run clean rebuild

all:
	@mkdir -p $(CMAKE_BUILD_DIR)
	cd $(CMAKE_BUILD_DIR) && cmake ..
	cd $(CMAKE_BUILD_DIR) && make -j8

run: all
	@./$(CMAKE_BUILD_DIR)/$(TARGET_NAME)

clean:
	rm -rf $(CMAKE_BUILD_DIR)

rebuild: clean all
