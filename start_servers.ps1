# Start backend server
Write-Host "Starting Backend Server..." -ForegroundColor Cyan
Start-Process -FilePath "C:\Program Files\nodejs\node.exe" -ArgumentList "d:\GIKI\Projects\P1\Customer_Churn_Prediction\backend\server.js" -NoNewWindow

# Wait for backend to start
Start-Sleep -Seconds 3

# Start frontend
Write-Host "Starting Frontend Server..." -ForegroundColor Cyan
Start-Process -FilePath "C:\Program Files\nodejs\node.exe" -ArgumentList "-e `"process.chdir('d:\\GIKI\\Projects\\P1\\Customer_Churn_Prediction\\frontend'); require('react-scripts/scripts/start.js')`"" -NoNewWindow

Write-Host "Both servers started!" -ForegroundColor Green
Write-Host "Backend: http://localhost:5000" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
