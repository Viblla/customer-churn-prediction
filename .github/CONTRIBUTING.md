# Contributing to Customer Churn Prediction Dashboard

Thank you for your interest in contributing! We welcome contributions from everyone.

## How to Contribute

### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Keep commits small and focused
- Write clear commit messages
- Test your changes locally

### 4. Install Dependencies

**For ML Model:**
```bash
cd ml_model
pip install -r requirements.txt
```

**For Backend:**
```bash
cd backend
npm install
```

**For Frontend:**
```bash
cd frontend
npm install
```

### 5. Test Your Changes

**Run backend tests:**
```bash
cd backend
npm test
```

**Run frontend tests:**
```bash
cd frontend
npm test
```

**Train and validate ML model:**
```bash
cd ml_model
python train_model.py
python predict.py
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a clear description of your changes.

## Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Keep code clean and readable

### Commit Messages
- Use descriptive commit messages
- Format: `type: description`
- Examples: `feat: add new ML model`, `fix: resolve API bug`, `docs: update README`

### Pull Request Process
1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Request review from maintainers
5. Address feedback and update PR

### Areas for Contribution
- **ML Models**: Improve existing models or add new algorithms
- **Backend**: Add new API endpoints or optimize existing ones
- **Frontend**: Enhance UI/UX or add new visualizations
- **Documentation**: Improve README, add tutorials, create guides
- **Bug Fixes**: Report and fix issues
- **Performance**: Optimize code and improve efficiency

## Report Issues

Found a bug or have a suggestion? Please open an issue with:
- Clear description of the problem
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment details

## Questions?

Feel free to open a discussion or contact the maintainers.

---

Thank you for contributing!
