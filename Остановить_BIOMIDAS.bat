@echo off
chcp 65001 > nul
title BIOMIDAS — Остановка

echo.
echo  Остановка BIOMIDAS...
docker compose down
echo.
echo  ✓ BIOMIDAS остановлен.
echo.
pause
