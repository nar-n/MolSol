# Simple installation script for PyTorch Geometric

# Get PyTorch version
$TORCH_VERSION = python -c "import torch; print(torch.__version__)"
Write-Host "Detected PyTorch version: $TORCH_VERSION"

# Install basic dependencies first
pip install numpy matplotlib networkx pandas scikit-learn scipy

# Try simplified PyTorch Geometric installation
Write-Host "Attempting to install PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric

# If you need the specific extensions, try one by one
Write-Host "Note: torch-scatter and other extensions might not be available for PyTorch $TORCH_VERSION"
Write-Host "You can try installing them separately if needed" -ForegroundColor Cyan
Write-Host "Your current GNN example will work without them!" -ForegroundColor Green

Write-Host "`nInstallation completed. You can now run: python main.py" -ForegroundColor Green
