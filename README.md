# AI Companion Installation Guide

This guide will help you set up a Python virtual environment, install dependencies, and run the AI companion script. Using Python 11 is recommended.

## 0. Clone this repository

```bash
git clone 'https://github.com/LumberJack14/ai-companion'
```

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
http://localhost:1234/
```

In the script you have to specify the model you're using:
```python
completion = client.chat.completions.create(
    model="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
    messages=history,
    temperature=0.7
)
```

## 4. Install ffmpeg and espeak-ng
Go to https://ffmpeg.org/download.html to install ffmpeg bianries. Add the bin folder to you PATH.

To install espeak-ng on Windows:

1. Go to espeak-ng releases: https://github.com/espeak-ng/espeak-ng/releases
2. Click on Latest release
3. Download the appropriate *.msi file (e.g. espeak-ng-20191129-b702b03-x64.msi)
4. Run the downloaded installer

## 5. Running the Script

```bash
python app.py
```

variable __character__ is responsible for your assistant's personality
```python
character = """
    Refer to the user as 'Master',
    You are his bitch, pretty much sums it up...
    You have the following traits: Argumentative, Honest, Logical, Polite, Feminine, Submissive, Concise.
    """
```