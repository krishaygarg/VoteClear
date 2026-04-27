# VoteClear

VoteClear is an intelligent political recommendation system that helps voters make informed decisions by matching their personal values and preferences with candidate stances on various policy issues. The system uses AI-powered analysis to provide personalized candidate recommendations based on practical, everyday concerns rather than abstract political ideologies.

## Features

- **Interactive Chat Interface**: Engage in natural conversation to explore your political preferences
- **AI-Powered Analysis**: Uses Google's Gemini AI for intelligent candidate matching
- **Multi-Election Support**: Currently supports 2025 Virginia Gubernatorial and NYC Mayoral elections
- **Research-Driven**: Built on comprehensive candidate research data with verifiable sources
- **Practical Focus**: Asks questions about real-world scenarios and everyday tradeoffs
- **Session Management**: Maintains conversation history for personalized recommendations

## Technology Stack

- **Backend**: Flask web framework
- **AI/ML**: LangChain with Google Gemini 2.5 Flash
- **Data Processing**: Pickle for storing candidate research data
- **Session Management**: Flask-Session for user conversation tracking
- **Web Search**: Tavily search integration for real-time information
- **Frontend**: HTML/CSS/JavaScript with responsive design

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishaygarg/VoteClear.git
   cd VoteClear
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. **Prepare data files**
   - Ensure the following pickle files exist in the project root:
     - `candidates.pkl` - List of confirmed candidates
     - `policyareas.pkl` - Predefined policy areas
     - `research.pkl` - Comprehensive candidate research data
   - Output files should be placed in `output/` directory:
     - `output1.txt` - Research data for election ID 1
     - `output2.txt` - Research data for election ID 2

## Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - Select an election from the available options
   - Begin chatting with the AI recommendation system

3. **Interaction Flow**
   - The system will ask you 5 multiple-choice questions about practical scenarios
   - Each question explores different dimensions of everyday life and preferences
   - After gathering sufficient information, it provides:
     - Holistic review of your alignment with each candidate
     - Summary of areas of alignment and tradeoffs
     - Personalized candidate recommendation

## Project Structure

```
VoteClear/
├── main.py                 # Main Flask application
├── politics.py            # Candidate research and analysis
├── politicsapi.py         # API integration for political data
├── questionasker.py       # Question generation logic
├── questionaskeragent.py  # AI agent for question asking
├── recommendationsystem.py # Core recommendation engine
├── tools.py               # Utility functions and search tools
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── templates/             # HTML templates
│   ├── elections.html     # Election selection page
│   └── chat.html          # Chat interface
├── static/                # Static assets (CSS, JS, images)
├── output/                # Research data files
├── flask_session/         # Session storage directory
└── data/                  # Pickle data files
    ├── candidates.pkl
    ├── policyareas.pkl
    └── research.pkl
```

## API Integration

The application integrates with several external APIs:

- **Google Gemini AI**: For natural language processing and candidate analysis
- **Tavily Search**: For real-time web search and information gathering
- **Wikipedia**: For additional context and background information

## Data Sources

Candidate research data is compiled from:
- Official campaign websites
- Public statements and interviews
- Voting records (when available)
- Policy position documents
- News articles and fact-checking sources

All sources are verified and included in the recommendation output for transparency.

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for Gemini AI integration
- `TAVILY_API_KEY`: Required for web search functionality

### Rate Limiting
The application includes built-in rate limiting for API calls:
- Gemini API: 9 requests per minute
- Configurable through `GEMINI_RPM_LIMIT` in `politics.py`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

VoteClear is designed to be an impartial tool for voter education. The recommendations generated are based on publicly available information and AI analysis. Users should consider this as one source of information among many when making voting decisions.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team through the repository

---

**Note**: This application requires active API keys for full functionality. Ensure all environment variables are properly configured before running.
