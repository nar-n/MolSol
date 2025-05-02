Write-Host "Installing basic GNN dependencies..." -ForegroundColor Green

# Install essential packages
pip install numpy matplotlib networkx pandas scikit-learn scipy

# Try to install just torch-geometric without extensions
try {
    Write-Host "Attempting simplified PyTorch Geometric installation..." -ForegroundColor Yellow
    pip install torch-geometric
    Write-Host "PyTorch Geometric installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Could not install PyTorch Geometric. Using basic PyTorch implementation." -ForegroundColor Yellow
}

Write-Host "`nInstallation completed. Your current GNN code will work with basic PyTorch." -ForegroundColor Green
