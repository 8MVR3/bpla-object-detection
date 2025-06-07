# Минимально необходимая версия CMake
cmake_minimum_required(VERSION 3.15)

# Название проекта
project(bpla-object-detection)

# ─── Подключение внешней библиотеки ZLIB через vcpkg ────────────────
# Требуем наличие установленного пакета ZLIB
find_package(ZLIB REQUIRED)

# ─── Создание основной исполняемой цели ─────────────────────────────
# Создаём бинарь с именем bpla_app из исходника main.cpp
add_executable(bpla_app main.cpp)

# ─── Линковка внешней библиотеки ZLIB с целью ──────────────────────
# Добавляем зависимость ZLIB на уровне target
target_link_libraries(bpla_app PRIVATE ZLIB::ZLIB)
