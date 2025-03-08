# EduParse

A command-line utility to convert scanned textbook pages into structured lessons with text, images, and questions.

## Features

- **OCR Text Extraction**: Extract text from scanned images while preserving layout
- **Image Detection**: Automatically identify and extract images from textbook pages
- **AI-Powered Lesson Generation**: Use Claude AI to create structured lessons
- **Image Positioning**: Intelligently place images in appropriate contexts
- **Question Generation**: Automatically generate relevant questions for the lesson
- **Multi-page Processing**: Process entire chapters and combine them into coherent lessons
- **Beautiful CLI Interface**: Rich, colorful console output with progress tracking

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/eduparse.git
   cd eduparse
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Ubuntu/Debian**:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - **macOS**:
     ```bash
     brew install tesseract
     ```
   - **Windows**:
     Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

4. Get a Claude API key from [Anthropic](https://www.anthropic.com/)

## Usage

### Basic Usage

Process a single textbook page:

```bash
python eduparse.py --input path/to/textbook_page.jpg --output lessons/ --api-key your_claude_api_key
```

Process multiple pages in a directory:

```bash
python eduparse.py --input path/to/textbook_pages/ --output lessons/ --api-key your_claude_api_key
```

### Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Path to input image or directory of images |
| `--output` | `-o` | Path to output directory |
| `--api-key` | `-k` | API key for Claude AI |
| `--model` | `-m` | Claude model to use (default: claude-3-5-sonnet-20240620) |
| `--format` | `-f` | Output format: markdown or html (default: markdown) |
| `--combine` | `-c` | Combine multiple pages into a single lesson |
| `--preview` | `-p` | Show a preview of the generated lesson |
| `--check` | | Check if all dependencies are installed |

### Examples

Check if all dependencies are installed:

```bash
python eduparse.py --check
```

Process a directory of pages and combine them into one lesson:

```bash
python eduparse.py -i textbook/chapter1/ -o lessons/ -k your_claude_api_key -c -p
```

Use a different Claude model and output in HTML format:

```bash
python eduparse.py -i textbook/page.jpg -o lessons/ -k your_claude_api_key -m claude-3-opus-20240229 -f html
```

## Output

The tool generates the following outputs:

1. **Lesson Content**: Markdown or HTML file with the structured lesson
2. **Extracted Images**: Stored in an 'images' subdirectory
3. **Metadata**: JSON file with processing information and image positions

## How It Works

1. **Text Extraction**: OCR is used to extract text while preserving layout information
2. **Image Detection**: Computer vision algorithms identify and extract images
3. **Claude Processing**: The text and images are sent to Claude AI with position information
4. **Lesson Generation**: Claude creates a well-structured lesson with proper image placement
5. **Output Formatting**: The content is saved in the desired format with images properly linked

## Requirements

- Python 3.7+
- Tesseract OCR
- Claude API key
- Sufficient disk space for extracted images

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 