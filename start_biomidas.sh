#!/bin/bash

echo ""
echo " ╔══════════════════════════════════════╗"
echo " ║         BIOMIDAS  v1.0               ║"
echo " ║   Анализ микробиомных данных 16S     ║"
echo " ╚══════════════════════════════════════╝"
echo ""

# Проверяем Docker
if ! command -v docker &> /dev/null; then
    echo " [ОШИБКА] Docker не найден."
    echo " Установите Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo " [ОШИБКА] Docker не запущен. Запустите Docker Desktop и попробуйте снова."
    exit 1
fi

echo " Остановка предыдущего запуска..."
docker compose down > /dev/null 2>&1

echo " Сборка и запуск BIOMIDAS..."
echo " (первый запуск может занять 3-5 минут)"
docker compose up --build -d

if [ $? -ne 0 ]; then
    echo " [ОШИБКА] Не удалось запустить приложение."
    exit 1
fi

echo " Ожидаем готовности сервера..."
sleep 8

echo " Открываем браузер..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501
elif command -v open &> /dev/null; then
    open http://localhost:8501
fi

echo ""
echo " ✓ BIOMIDAS запущен: http://localhost:8501"
echo " Для остановки: docker compose down"
echo ""
