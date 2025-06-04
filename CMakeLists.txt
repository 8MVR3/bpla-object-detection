cmake_minimum_required(VERSION 3.15)
project(bpla-object-detection)

# Подключаем ZLIB из vcpkg
find_package(ZLIB REQUIRED)

# ✅ Название цели изменено с main -> bpla_app
add_executable(bpla_app main.cpp)

# Линкуем ZLIB
target_link_libraries(bpla_app PRIVATE ZLIB::ZLIB)
