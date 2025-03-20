📦 Project Setup & Installation

This guide will help you quickly set up your Python project, including creating a virtual environment, installing dependencies, and running the application.

✅ Step 1: Clone the Repository

Clone the repository from GitHub:

git clone <git@github.com:nankam1803/ragModelOllama.git>
cd <ragModelOllama>


✅ Step 2: Create and Activate a Virtual Environment

Create a virtual environment named .venv:

Windows:

python -m venv .venv
.\.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

✅ Step 3: Install Dependencies

Install all the required Python libraries using the provided requirements.txt file:

pip install --upgrade pip
pip install -r requirements.txt

✅ Step 4: Run the Application

Run your Python script or application:

python ragModel.py

✅ Step 5: Change PDF and Question

To customize the PDF file and the question you're asking, edit the Python script ragModel.py:

local_path = r"your/path/to/your_file.pdf"  # Replace with your PDF path
question = "Your custom question here"  # Replace with your question

🚩 Troubleshooting

If you encounter errors related to Poppler (common when working with PDFs), follow these additional steps:

Windows Users:

Download Poppler.
Extract it to a folder (e.g., C:\poppler).
Add C:\poppler\Library\bin to your system PATH.

macOS Users:
brew install poppler

Linux Users (Ubuntu/Debian):
sudo apt-get install poppler-utils

Then, restart your terminal and re-run your application.