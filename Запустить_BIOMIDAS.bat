@echo off
chcp 65001 > nul
title BIOMIDAS — Запуск

echo.
echo  ╔══════════════════════════════════════╗
echo  ║         BIOMIDAS  v1.0               ║
echo  ║   Анализ микробиомных данных 16S     ║
echo  ╚══════════════════════════════════════╝
echo.

:: Проверяем наличие Docker
docker --version > nul 2>&1
if %errorlevel% neq 0 (
    echo  [ОШИБКА] Docker не найден.
    echo.
    echo  Установите Docker Desktop:
    echo  https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

:: Проверяем, запущен ли Docker
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo  [ОШИБКА] Docker Desktop не запущен.
    echo  Запустите Docker Desktop и попробуйте снова.
    echo.
    pause
    exit /b 1
)

:: Останавливаем старый контейнер если есть
echo  Подготовка...
docker compose down > nul 2>&1

:: Сборка и запуск
echo  Сборка и запуск BIOMIDAS...
echo  (первый запуск может занять 3-5 минут)
echo.
docker compose up --build -d

if %errorlevel% neq 0 (
    echo.
    echo  [ОШИБКА] Не удалось запустить приложение.
    echo  Проверьте подключение к интернету и попробуйте снова.
    pause
    exit /b 1
)

:: Ждём готовности
echo  Ожидаем готовности сервера...
timeout /t 8 /nobreak > nul

:: Открываем браузер
echo  Открываем BIOMIDAS в браузере...
start http://localhost:8501

echo.
echo  ✓ BIOMIDAS запущен: http://localhost:8501
echo.
echo  Для остановки запустите: Остановить_BIOMIDAS.bat
echo  Или нажмите любую клавишу — браузер останется открытым.
echo.
pause
