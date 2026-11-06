@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title RAG Console - Knowledge Base Q and A

echo ============================================================
echo   RAG Console  -  Q/A  +  Parameters / Metrics
echo ============================================================
echo.

rem ---------- 1. python runtime ----------
if not exist ".venv\Scripts\python.exe" (
  echo [X] Virtual environment not found:
  echo     .venv\Scripts\python.exe
  echo.
  echo     Create it once with:
  echo       python -m venv .venv
  echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

rem ---------- 2. vector store ----------
if not exist "vector_store" (
  echo [!] vector_store\ not found - run:  python build_index.py
  echo.
)

rem ---------- 3. Ollama ----------
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 3 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
  echo [!] Ollama is NOT reachable at http://localhost:11434
  echo     The page will still open, but Q/A will fail until Ollama is running.
  echo     Start it from the tray icon, or run:  ollama serve
  echo.
  pause
) else (
  echo [OK] Ollama is running.
)

rem ---------- 4. port ----------
set PORT=8501
powershell -NoProfile -Command "try { Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo [!] Port 8501 is busy - switching to 8502
  set PORT=8502
)

rem ---------- 5. urls ----------
echo.
echo ------------------------------------------------------------
".venv\Scripts\python.exe" show_urls.py %PORT%
echo ------------------------------------------------------------
echo   Open this address in your browser:
echo       http://localhost:%PORT%
echo.
echo   (No browser is opened automatically - avoids the
echo    "Edge is running with elevated privileges" dialog
echo    when this window runs as Administrator.)
echo.
echo   Close this window or press Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false --logger.hideWelcomeMessage true

echo.
echo ------------------------------------------------------------
echo Streamlit stopped.
pause
