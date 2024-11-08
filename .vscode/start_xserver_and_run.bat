@echo off
REM Check if VcXsrv is already running
tasklist /FI "IMAGENAME eq vcxsrv.exe" 2>NUL | find /I /N "vcxsrv.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo VcXsrv is already running.
) else (
    echo Starting VcXsrv...
    start "" "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -multiwindow -clipboard -wgl -ac
    timeout /t 5 >nul
)

REM Set DISPLAY environment variable
set DISPLAY=host.docker.internal:0.0

REM Run the Docker container
docker-compose up --build

REM Optionally, you can run the container directly without docker-compose
REM docker run --rm -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix your_docker_image