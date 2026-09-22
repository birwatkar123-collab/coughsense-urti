@echo off
set "REVIEW=%~dp0urti-audio-reviews.json"

if not exist "%REVIEW%" (
  echo Review file not found: %REVIEW%
  echo.
  echo 1. Open D:\urti\experiments\listening-review\index.html in a browser
  echo 2. Complete all 90 recordings manually
  echo 3. Click Download reviews
  echo 4. Save/copy the exported file to: %REVIEW%
  echo 5. Rerun this script, or run:
  echo    python analyze_reviews.py --review "%REVIEW%"
  exit /b 2
)

python "%~dp0analyze_reviews.py" --review "%REVIEW%"
exit /b %ERRORLEVEL%