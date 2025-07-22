# AI Companion Installation Guide

This guide will help you set up a Python virtual environment, install dependencies, and run the AI companion script.

---

## 1. Create a Python Virtual Environment

It's best practice to isolate your project dependencies using a virtual environment.

Open a terminal or command prompt in your project folder, then run:

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Set Up LM Studio API
Make sure your local LM Studio server is running and accessible at:

```bash
http://localhost:1234/v1
```

## 4. Running the Script

```bash
python app.py
```