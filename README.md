# AutoGen Literature Review Assistant

## Overview
The AutoGen Literature Review Assistant is a web application built using Streamlit that leverages multi-agent AI systems to assist researchers in conducting literature reviews. The application allows users to search for academic papers on arXiv and generate comprehensive summaries of the selected papers.

## Features
- Search for academic papers on arXiv based on user-defined topics.
- Generate structured literature reviews with detailed summaries.
- User-friendly interface powered by Streamlit.
- Real-time updates during the literature review process.

## Project Structure
```
autogen-literature-review/
├── src/
│   └── main.py                # Main application logic
├── .env                        # Environment variables (API keys, etc.)
├── .gitignore                  # Files and directories to ignore by Git
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd autogen-literature-review
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Create a `.env` file in the root directory and add your API keys:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run src/main.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the AutoGen Literature Review Assistant.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.