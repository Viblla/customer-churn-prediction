@echo off
echo Killing any existing node processes...
taskkill /F /IM node.exe 2>nul

timeout /T 2 /nobreak

echo Starting Backend Server...
cd /d "d:\GIKI\Projects\P1\Customer_Churn_Prediction\backend"
start "Backend Server" /D "d:\GIKI\Projects\P1\Customer_Churn_Prediction\backend" node.exe server.js

timeout /T 3 /nobreak

echo Backend should be running on http://localhost:5000
echo.
echo Starting Frontend Server...
cd /d "d:\GIKI\Projects\P1\Customer_Churn_Prediction\frontend"
start "Frontend Server" /D "d:\GIKI\Projects\P1\Customer_Churn_Prediction\frontend" cmd.exe /k npm start

echo.
echo All servers started!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
