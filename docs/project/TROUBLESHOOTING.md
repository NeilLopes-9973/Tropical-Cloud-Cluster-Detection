# ITCC Project - Troubleshooting Guide

## Dashboard Issues

### Issue: "localhost not working" or "Cannot connect to localhost:8504"

**Solution 1: Start the Dashboard**
```bash
# Double-click this file:
START_DASHBOARD.bat

# OR run manually:
cd 04_dashboards
python -m streamlit run itcc_dashboard.py
```

**Solution 2: Check if Port is Already in Use**
```bash
# Try a different port
cd 04_dashboards
python -m streamlit run itcc_dashboard.py --server.port 8505
```

**Solution 3: Check if Streamlit is Installed**
```bash
pip install streamlit plotly pandas numpy
```

---

## Common Errors and Fixes

### Error: "ModuleNotFoundError: No module named 'streamlit'"

**Fix:**
```bash
pip install streamlit plotly pandas numpy
```

### Error: "FileNotFoundError: data/final/ITCC_v2.csv"

**Fix:**
```bash
cd 01_data_preparation
python build_itcc_dataset.py
python enhance_itcc_dataset.py
```

### Error: "Port 8504 is already in use"

**Fix 1: Kill existing process**
```bash
# Find process using port
netstat -ano | findstr "8504"

# Kill process (replace PID with actual number)
taskkill /PID <PID> /F
```

**Fix 2: Use different port**
```bash
python -m streamlit run itcc_dashboard.py --server.port 8505
```

### Error: "streamlit: command not found"

**Fix:**
```bash
# Use Python module syntax
python -m streamlit run itcc_dashboard.py
```

---

## Dashboard Not Loading Data

### Check 1: Verify Data File Exists
```bash
# Should see ITCC_v2.csv
dir data\final\ITCC_v2.csv
```

### Check 2: Rebuild Dataset
```bash
cd 01_data_preparation
python build_itcc_dataset.py
python enhance_itcc_dataset.py
```

### Check 3: Check File Path
- Dashboard looks for: `data/final/ITCC_v2.csv`
- Make sure you're running from project root or `04_dashboards/`

---

## API Not Working

### Issue: "API Offline" in dashboard

**Fix:**
```bash
cd api
python -m uvicorn main:app --port 8081 --reload
```

### Check API Status
```bash
# Open in browser:
http://localhost:8081/docs
```

---

## Browser Issues

### Dashboard Opens but Shows Blank Page

**Fix 1: Clear Browser Cache**
- Press Ctrl+Shift+Delete
- Clear cache and reload

**Fix 2: Try Different Browser**
- Chrome (recommended)
- Firefox
- Edge

**Fix 3: Disable Extensions**
- Ad blockers may interfere
- Try incognito/private mode

---

## Performance Issues

### Dashboard is Slow

**Fix 1: Reduce Data Range**
- Use date/time filters
- Select specific lifecycle stage
- Limit Tb_min range

**Fix 2: Close Other Applications**
- Free up RAM
- Close unused browser tabs

**Fix 3: Restart Dashboard**
```bash
# Press Ctrl+C to stop
# Then restart:
python -m streamlit run itcc_dashboard.py
```

---

## Installation Issues

### pip install fails

**Fix 1: Update pip**
```bash
python -m pip install --upgrade pip
```

**Fix 2: Install one by one**
```bash
pip install streamlit
pip install plotly
pip install pandas
pip install numpy
```

**Fix 3: Use conda (if available)**
```bash
conda install streamlit plotly pandas numpy
```

---

## Quick Diagnostics

### Check Python Version
```bash
python --version
# Should be 3.8 or higher
```

### Check Installed Packages
```bash
pip list | findstr streamlit
pip list | findstr plotly
pip list | findstr pandas
```

### Check Current Directory
```bash
cd
# Should be in Miniproject_tcc or 04_dashboards
```

---

## Still Not Working?

### Step-by-Step Reset

1. **Close all terminals and browsers**

2. **Reinstall dependencies**
```bash
pip uninstall streamlit plotly pandas -y
pip install streamlit plotly pandas numpy
```

3. **Navigate to correct folder**
```bash
cd d:\Miniproject_tcc\04_dashboards
```

4. **Start fresh**
```bash
python -m streamlit run itcc_dashboard.py
```

5. **Open browser manually**
- Go to: http://localhost:8501
- Or: http://localhost:8504

---

## Contact Information

If issues persist:
1. Check folder-specific README files
2. Review PROJECT_COMPLETE_FLOW.md
3. Verify all files are in correct locations

---

## Quick Reference

### Start Dashboard (Easy Way)
```bash
# Just double-click:
START_DASHBOARD.bat
```

### Start Dashboard (Manual)
```bash
cd 04_dashboards
python -m streamlit run itcc_dashboard.py
```

### Stop Dashboard
```
Press Ctrl+C in terminal
```

### Check What's Running
```bash
netstat -ano | findstr "850"
```

---

**Last Updated**: October 2025  
**Version**: 1.0
