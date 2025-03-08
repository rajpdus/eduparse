# EduParse Architecture

This document outlines the architecture of the EduParse project, explaining how the different components interact to convert scanned textbook pages into structured lessons.

## System Overview

EduParse follows a pipeline architecture where data flows through several processing stages:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │     │             │
│  Input      │────▶│  OCR Text   │────▶│  Image      │────▶│  Claude AI  │────▶│  Output     │
│  Processing │     │  Extraction │     │  Extraction │     │  Processing │     │  Generation │
│             │     │             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Component Architecture

### 1. Input Processing

**Purpose**: Handle input files (single images or directories) and prepare them for processing.

**Key Functions**:
- `process_textbook_page()`: Process a single textbook page
- `process_multiple_pages()`: Process multiple pages from a directory

**Flow Diagram**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  CLI Args   │────▶│  Validate   │────▶│  Load       │
│  Parsing    │     │  Inputs     │     │  Images     │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 2. OCR Text Extraction

**Purpose**: Extract text from scanned images while preserving layout information.

**Key Functions**:
- `preprocess_image()`: Prepare images for OCR processing
- `extract_text_with_layout()`: Extract text with position information

**Flow Diagram**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Image      │────▶│  Convert to │────▶│  Apply      │────▶│  OCR        │
│  Loading    │     │  Grayscale  │     │  Threshold  │     │  Processing │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐
                                                            │             │
                                                            │  Text with  │
                                                            │  Layout     │
                                                            │             │
                                                            └─────────────┘
```

### 3. Image Extraction

**Purpose**: Detect and extract images embedded within the textbook pages.

**Key Functions**:
- `detect_and_extract_images()`: Identify and extract images from the page

**Flow Diagram**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Convert to │────▶│  Apply      │────▶│  Find       │────▶│  Filter     │
│  Binary     │     │  Threshold  │     │  Contours   │     │  Contours   │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐     ┌─────────────┐
                                                            │             │     │             │
                                                            │  Extract    │────▶│  Save       │
                                                            │  Images     │     │  Images     │
                                                            │             │     │             │
                                                            └─────────────┘     └─────────────┘
```

### 4. Claude AI Processing

**Purpose**: Use Claude AI to generate structured lessons from the extracted text and images.

**Key Functions**:
- `generate_lesson_with_claude()`: Send text and images to Claude and process the response

**Flow Diagram**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Encode     │────▶│  Create     │────▶│  Send to    │────▶│  Process    │
│  Images     │     │  Prompt     │     │  Claude API │     │  Response   │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 5. Output Generation

**Purpose**: Save the generated lesson and associated metadata.

**Key Functions**:
- `combine_lessons()`: Optionally combine multiple lessons into one
- `display_preview()`: Show a preview of the generated lesson

**Flow Diagram**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Format     │────▶│  Save       │────▶│  Generate   │
│  Content    │     │  Files      │     │  Metadata   │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Data Flow

The following diagram illustrates the complete data flow through the system:

```
┌─────────────┐
│             │
│  Scanned    │
│  Textbook   │
│  Page       │
│             │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│             │     │             │
│  OCR Text   │     │  Image      │
│  Extraction │     │  Extraction │
│             │     │             │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│                                 │
│  Claude AI Processing           │
│  - Structure text               │
│  - Position images              │
│  - Generate questions           │
│                                 │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│                                 │
│  Output                         │
│  - Markdown/HTML lesson         │
│  - Extracted images             │
│  - Metadata                     │
│                                 │
└─────────────────────────────────┘
```

## Component Dependencies

```
┌─────────────┐
│             │
│  Rich       │──┐
│  (CLI)      │  │
│             │  │
└─────────────┘  │
                 │
┌─────────────┐  │
│             │  │
│  OpenCV     │──┤
│  (Images)   │  │
│             │  │
└─────────────┘  │     ┌─────────────┐
                 │     │             │
┌─────────────┐  │     │  EduParse   │
│             │  └────▶│  Core       │
│  Tesseract  │──┐     │  Logic      │
│  (OCR)      │  │     │             │
│             │  │     └─────────────┘
└─────────────┘  │
                 │
┌─────────────┐  │
│             │  │
│  Anthropic  │──┘
│  (Claude)   │
│             │
└─────────────┘
```

## Design Patterns

1. **Pipeline Pattern**: Data flows through a series of processing stages
2. **Command Pattern**: CLI arguments determine the execution flow
3. **Strategy Pattern**: Different processing strategies based on input type
4. **Facade Pattern**: Simple interface hiding complex subsystems

## Extension Points

EduParse is designed to be extensible in several ways:

1. **Alternative OCR Engines**: The OCR component can be replaced with other engines
2. **Different LLM Providers**: The Claude integration can be extended to support other LLMs
3. **Custom Image Processing**: The image detection algorithms can be enhanced or replaced
4. **Additional Output Formats**: New output formats beyond Markdown and HTML can be added

## Future Architecture Considerations

1. **Microservice Architecture**: Split components into separate services for scalability
2. **API Layer**: Add a REST API for programmatic access
3. **Database Integration**: Store processed lessons and metadata in a database
4. **Web Interface**: Add a web-based UI for easier interaction
5. **Parallel Processing**: Implement parallel processing for multiple pages 