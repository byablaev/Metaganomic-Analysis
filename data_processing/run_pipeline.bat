@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

set PIPELINE_DIR=%~dp0
set IMAGE_NAME=qiime2-biomidas-pipeline
set CONTAINER_NAME=qiime2_biomidas_pipeline
set FASTQ_DIR=%PIPELINE_DIR%fastq
set OUTPUT_DIR=%PIPELINE_DIR%output
set SILVA_CACHE=%PIPELINE_DIR%silva_cache

docker info >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Docker не запущен.
    echo Запустите Docker Desktop и повторите.
    pause & exit /b 1
)

if not exist "%FASTQ_DIR%"   mkdir "%FASTQ_DIR%"
if not exist "%OUTPUT_DIR%"  mkdir "%OUTPUT_DIR%"
if not exist "%SILVA_CACHE%" mkdir "%SILVA_CACHE%"

:: Проверить наличие FASTQ-файлов
set FASTQ_COUNT=0
for %%f in ("%FASTQ_DIR%\*.fastq.gz") do set /a FASTQ_COUNT+=1

:: Проверить наличие metadata.csv
set META_OK=0
if exist "%PIPELINE_DIR%metadata.csv" set META_OK=1

echo.
echo  16S rRNA пайплайн (QIIME2 + SILVA 138 -^> BIOMIDAS)
echo  FASTQ  : %FASTQ_DIR%  [!FASTQ_COUNT! файлов]
echo  Вывод  : %OUTPUT_DIR%
if !META_OK!==0 (
    echo  ВНИМАНИЕ: metadata.csv не найден! Создайте его перед запуском.
    echo            Пример: metadata_example.csv
)
echo.
echo  [1] Собрать образ + запустить
echo  [2] Запустить (образ уже есть)
echo  [3] Только собрать образ
echo  [4] Логи
echo  [5] Открыть результаты
echo  [6] Выход
echo.
set /p CHOICE="Выбор: "

if "%CHOICE%"=="1" goto BUILD_AND_RUN
if "%CHOICE%"=="2" goto RUN
if "%CHOICE%"=="3" goto BUILD
if "%CHOICE%"=="4" goto LOGS
if "%CHOICE%"=="5" goto OPEN
if "%CHOICE%"=="6" goto END
echo Неверный ввод. & pause & goto END

:BUILD
docker build -t %IMAGE_NAME%:latest "%PIPELINE_DIR%"
if errorlevel 1 ( echo Ошибка сборки. & pause & exit /b 1 )
echo Образ собран.
pause & goto END

:BUILD_AND_RUN
docker build -t %IMAGE_NAME%:latest "%PIPELINE_DIR%"
if errorlevel 1 ( echo Ошибка сборки. & pause & exit /b 1 )

:RUN
if !FASTQ_COUNT!==0 (
    echo ОШИБКА: нет .fastq.gz файлов в %FASTQ_DIR%
    echo Скопируйте FASTQ-файлы и повторите.
    pause & goto END
)
if !META_OK!==0 (
    echo ОШИБКА: файл metadata.csv не найден в %PIPELINE_DIR%
    echo Создайте его по образцу metadata_example.csv и повторите.
    pause & goto END
)

docker rm -f %CONTAINER_NAME% >nul 2>&1

echo.
echo Запуск... DADA2 занимает 30-90 мин. Не закрывайте окно.
echo.

docker run --rm ^
    --name %CONTAINER_NAME% ^
    -v "%FASTQ_DIR%:/data/fastq:ro" ^
    -v "%OUTPUT_DIR%:/pipeline/output" ^
    -v "%SILVA_CACHE%:/pipeline/silva_cache" ^
    -e FASTQ_DIR=/data/fastq ^
    -e OUTPUT_DIR=/pipeline/output ^
    -e SILVA_CLASSIFIER=/pipeline/silva-138-99-seqs-515-806.qza ^
    -e METADATA_FILE=/pipeline/metadata.csv ^
    -e N_THREADS=8 ^
    -e TRUNC_LEN_F=250 ^
    -e TRUNC_LEN_R=200 ^
    -e MIN_FREQUENCY=10 ^
    -e MIN_SAMPLES=2 ^
    --memory=16g ^
    %IMAGE_NAME%:latest

if errorlevel 1 ( echo Ошибка пайплайна. Смотрите логи выше. & pause & exit /b 1 )

echo.
echo Готово! Файлы для BIOMIDAS:
echo   %PIPELINE_DIR%output\exported\biomidas\otu_table.csv
echo   %PIPELINE_DIR%output\exported\biomidas\taxonomy.csv
echo   %PIPELINE_DIR%output\exported\biomidas\metadata.csv
echo.
set /p OPEN="Открыть папку? (y/n): "
if /i "%OPEN%"=="y" explorer "%PIPELINE_DIR%output\exported\biomidas"
pause & goto END

:LOGS
if not exist "%PIPELINE_DIR%output\logs" (
    echo Логов нет — сначала запустите пайплайн. & pause & goto END
)
dir /b "%PIPELINE_DIR%output\logs\*.log"
echo.
set /p LOGNAME="Имя файла: "
if exist "%PIPELINE_DIR%output\logs\!LOGNAME!" (
    type "%PIPELINE_DIR%output\logs\!LOGNAME!" | more
) else ( echo Файл не найден. )
pause & goto END

:OPEN
if exist "%PIPELINE_DIR%output\exported\biomidas" (
    explorer "%PIPELINE_DIR%output\exported\biomidas"
) else if exist "%PIPELINE_DIR%output" (
    explorer "%PIPELINE_DIR%output"
) else ( echo Папка output не найдена. )
pause & goto END

:END
endlocal
