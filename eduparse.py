#!/usr/bin/env python3
"""
EduParse - Convert scanned textbook pages into structured lessons with text, images, and questions.

This utility uses OCR to extract text, computer vision to extract images,
and Claude AI to generate well-structured educational content.
"""

import os
import sys
import argparse
import base64
from pathlib import Path
import json
import time
import tempfile
from typing import List, Dict, Any, Tuple

# Rich imports for beautiful CLI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

# Computer vision and OCR
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

# Claude API
import anthropic

# Initialize Rich console
console = Console()

def encode_image(image_path: str) -> str:
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess the image for OCR"""
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get black and white image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return thresh, img

def extract_text_with_layout(image_path: str, progress=None) -> dict:
    """Extract text while preserving layout information"""
    if progress:
        progress.update(task_id, description="[bold yellow]Extracting text with OCR...[/bold yellow]")
    
    # Preprocess image
    preprocessed_img, _ = preprocess_image(image_path)
    
    # Use pytesseract's data frame output to get text with position information
    data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
    
    # Filter out empty text entries
    valid_indices = [i for i, text in enumerate(data['text']) if text and text.strip()]
    filtered_data = {key: [data[key][i] for i in valid_indices] for key in data}
    
    # Organize by paragraph and line
    text_blocks = []
    current_line = ""
    prev_line_num = -1
    
    for i in range(len(filtered_data['text'])):
        line_num = filtered_data['line_num'][i]
        
        if line_num != prev_line_num:
            if current_line:
                text_blocks.append(current_line)
                current_line = ""
            prev_line_num = line_num
        
        current_line += filtered_data['text'][i] + " "
    
    if current_line:
        text_blocks.append(current_line)
    
    full_text = "\n".join(text_blocks)
    
    return {
        'raw_data': filtered_data,
        'text_blocks': text_blocks,
        'full_text': full_text
    }

def detect_and_extract_images(image_path: str, output_dir: str, progress=None) -> List[Dict[str, Any]]:
    """Detect and extract images from the textbook page"""
    if progress:
        progress.update(task_id, description="[bold yellow]Detecting and extracting images...[/bold yellow]")
    
    # Load image
    original_img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    extracted_images = []
    
    # Filter and process contours
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 5000]
    
    for i, contour in enumerate(valid_contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the image
        image = original_img[y:y+h, x:x+w]
        
        # Save to disk
        image_path = os.path.join(images_dir, f"{base_name}_image_{i}.png")
        cv2.imwrite(image_path, image)
        
        # Calculate normalized position (as percentage of page dimensions)
        height, width = original_img.shape[:2]
        normalized_position = {
            'x_percent': round(x / width * 100, 2),
            'y_percent': round(y / height * 100, 2),
            'width_percent': round(w / width * 100, 2),
            'height_percent': round(h / height * 100, 2)
        }
        
        extracted_images.append({
            'path': image_path,
            'relative_path': f"images/{base_name}_image_{i}.png",
            'position': (x, y, w, h),
            'normalized_position': normalized_position
        })
    
    if progress:
        progress.update(task_id, 
                       description=f"[bold green]Found {len(extracted_images)} images[/bold green]")
    
    return extracted_images

def generate_lesson_with_claude(text_data: dict, extracted_images: List[Dict], 
                               api_key: str, model: str, progress=None) -> str:
    """Send text and images to Claude to generate a lesson"""
    if progress:
        progress.update(task_id, description="[bold magenta]Generating lesson with Claude...[/bold magenta]")
    
    # Encode the extracted images for Claude
    encoded_images = []
    for img_data in extracted_images:
        encoded_images.append({
            'data': encode_image(img_data['path']),
            'position': img_data['normalized_position'],
            'relative_path': img_data['relative_path']
        })
    
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """You are an expert educator and content creator.
    Your task is to convert textbook content into well-structured lessons.
    
    You will receive:
    1. The full text extracted from a textbook page
    2. Images extracted from that page with their position information
    
    Your job is to:
    1. Create a well-structured lesson from the text
    2. Insert the provided images at appropriate positions in the content
    3. Add captions and references to the images where appropriate
    4. Generate 3-5 thoughtful questions based on the material
    5. Format everything in clean, structured markdown with proper image placement"""
    
    # Build the message with text and all extracted images
    message_content = [
        {
            "type": "text",
            "text": f"""Here is the extracted text from a textbook page:
            
{text_data['full_text']}

I've also extracted {len(extracted_images)} images from this page. 
Each image has position information (x, y, width, height as percentages of the page dimensions).
Please place these images appropriately in your lesson based on their positions in the original text:

{json.dumps([{
    'image_id': i,
    'position': img['normalized_position']
} for i, img in enumerate(encoded_images)], indent=2)}

Create a well-structured lesson with these images properly positioned."""
        }
    ]
    
    # Add each image to the message content
    for i, img in enumerate(encoded_images):
        message_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img['data']
            }
        })
        
        # Add separator text after each image (except the last)
        if i < len(encoded_images) - 1:
            message_content.append({
                "type": "text",
                "text": f"Above is image {i} with position {img['position']}."
            })
    
    # Send to Claude
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        )
        
        # Extract the lesson content
        lesson_content = message.content[0].text
        
        # Process the returned content to replace image references
        for i, img in enumerate(extracted_images):
            # Replace various forms of image references
            placeholder_patterns = [
                f"![Image {i}]",
                f"![image {i}]",
                f"![Figure {i}]",
                f"![figure {i}]",
                f"Image {i}:"
            ]
            
            replacement = f"![Image {i}]({img['relative_path']})"
            
            for pattern in placeholder_patterns:
                lesson_content = lesson_content.replace(pattern, replacement)
        
        return lesson_content
        
    except Exception as e:
        if progress:
            progress.update(task_id, description=f"[bold red]Error calling Claude: {str(e)}[/bold red]")
        console.print(f"[bold red]Error calling Claude API: {str(e)}[/bold red]")
        return None

def process_textbook_page(image_path: str, output_dir: str, api_key: str, 
                         model: str = "claude-3-5-sonnet-20240620", 
                         output_format: str = 'markdown') -> str:
    """Process a textbook page with both OCR and image extraction, then use Claude for lesson generation"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        global task_id
        task_id = progress.add_task(f"[cyan]Processing {os.path.basename(image_path)}...[/cyan]", total=5)
        
        # 1. Extract text with layout information
        text_data = extract_text_with_layout(image_path, progress)
        progress.update(task_id, advance=1)
        
        # 2. Extract images
        extracted_images = detect_and_extract_images(image_path, output_dir, progress)
        progress.update(task_id, advance=1)
        
        # 3. Generate lesson with Claude
        lesson_content = generate_lesson_with_claude(text_data, extracted_images, api_key, model, progress)
        progress.update(task_id, advance=1)
        
        if not lesson_content:
            progress.update(task_id, description="[bold red]Failed to generate lesson[/bold red]")
            return None
        
        # 4. Save the lesson
        progress.update(task_id, description="[bold blue]Saving lesson content...[/bold blue]")
        
        if output_format == 'markdown':
            output_path = os.path.join(output_dir, f"{base_name}.md")
        else:  # HTML
            output_path = os.path.join(output_dir, f"{base_name}.html")
        
        with open(output_path, "w") as f:
            f.write(lesson_content)
        
        # 5. Save metadata
        metadata = {
            "source": str(image_path),
            "images": [{
                "path": img['relative_path'],
                "position": img['normalized_position']
            } for img in extracted_images],
            "model": model,
            "format": output_format,
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_dir, f"{base_name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        progress.update(task_id, advance=1)
        progress.update(task_id, description=f"[bold green]Successfully saved lesson to {output_path}[/bold green]")
        
        return output_path

def process_multiple_pages(input_dir: str, output_dir: str, api_key: str, 
                          model: str, output_format: str, combine: bool = False) -> List[str]:
    """Process multiple textbook pages and optionally combine them"""
    # Get all image files
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_files.append(os.path.join(input_dir, file))
    
    # Sort files to maintain page order
    image_files.sort()
    
    if not image_files:
        console.print("[bold red]No valid image files found in the input directory[/bold red]")
        return []
    
    console.print(f"[bold]Found {len(image_files)} image file(s) to process[/bold]")
    
    # Process each page
    processed_files = []
    
    for img_path in image_files:
        output_path = process_textbook_page(img_path, output_dir, api_key, model, output_format)
        if output_path:
            processed_files.append(output_path)
    
    if combine and len(processed_files) > 1:
        console.print("\n[bold yellow]Combining multiple pages into a single lesson...[/bold yellow]")
        combined_path = combine_lessons(processed_files, output_dir, api_key, model, output_format)
        return [combined_path]
    
    return processed_files

def combine_lessons(lesson_files: List[str], output_dir: str, api_key: str, 
                   model: str, output_format: str) -> str:
    """Combine multiple lesson files into a single lesson"""
    
    # Read all lesson contents
    all_lessons = []
    for file_path in lesson_files:
        with open(file_path, 'r') as f:
            lesson_content = f.read()
            all_lessons.append(lesson_content)
    
    combined_content = "\n\n---\n\n".join(all_lessons)
    
    with console.status("[bold green]Synthesizing combined lesson with Claude...[/bold green]"):
        # Use Claude to synthesize the content
        client = anthropic.Anthropic(api_key=api_key)
        
        system_prompt = """You are an expert educator. You'll receive content from multiple pages of a textbook that
        have been processed individually. Your task is to combine them into a single coherent lesson,
        removing redundancies, ensuring a logical flow, and maintaining all important content including images."""
        
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Here are multiple pages from a textbook that need to be combined into a single lesson:\n\n{combined_content}"
                    }
                ]
            )
            
            final_content = message.content[0].text
            
            # Save the combined lesson
            if output_format == 'markdown':
                combined_path = os.path.join(output_dir, "combined_lesson.md")
            else:
                combined_path = os.path.join(output_dir, "combined_lesson.html")
            
            with open(combined_path, "w") as f:
                f.write(final_content)
            
            console.print(f"[bold green]Successfully created combined lesson at {combined_path}[/bold green]")
            return combined_path
            
        except Exception as e:
            console.print(f"[bold red]Error combining lessons: {str(e)}[/bold red]")
            return None

def display_preview(file_path: str) -> None:
    """Display a preview of the generated lesson"""
    if not os.path.exists(file_path):
        console.print(f"[bold red]File not found: {file_path}[/bold red]")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Get first 500 characters for preview
    preview = content[:500] + ("..." if len(content) > 500 else "")
    
    console.print("\n[bold]Lesson Preview:[/bold]")
    console.print(Panel(Markdown(preview), title=os.path.basename(file_path)))
    console.print(f"\nFull lesson saved to: [bold cyan]{file_path}[/bold cyan]")

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    try:
        # Check Tesseract
        version = pytesseract.get_tesseract_version()
        console.print(f"[green]✓ Tesseract OCR version {version} found[/green]")
    except Exception:
        console.print("[bold red]✗ Tesseract OCR not found. Please install it:[/bold red]")
        console.print("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        console.print("  - macOS: brew install tesseract")
        console.print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    try:
        # Check OpenCV
        cv_version = cv2.__version__
        console.print(f"[green]✓ OpenCV version {cv_version} found[/green]")
    except Exception:
        console.print("[bold red]✗ OpenCV not found. Please install it: pip install opencv-python[/bold red]")
        return False
    
    return True

def validate_api_key(api_key: str) -> bool:
    """Validate Claude API key by making a simple request"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Simple validation request
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=10,
            temperature=0,
            messages=[
                {"role": "user", "content": "Hello, just validating my API key."}
            ]
        )
        console.print("[green]✓ Claude API key validated successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[bold red]✗ Invalid Claude API key: {str(e)}[/bold red]")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="""EduParse - Convert scanned textbook pages into structured lessons with text, images, and questions.
                    This utility uses OCR to extract text, computer vision to extract images,
                    and Claude AI to generate well-structured educational content."""
    )
    
    parser.add_argument('--input', '-i', required=True, 
                      help='Path to input image or directory of images')
    parser.add_argument('--output', '-o', required=True, 
                      help='Path to output directory')
    parser.add_argument('--api-key', '-k', required=True, 
                      help='API key for Claude AI')
    parser.add_argument('--model', '-m', 
                      choices=['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-5-sonnet-20240620', 'claude-3-opus-20240229'],
                      default='claude-3-5-sonnet-20240620', 
                      help='Claude model to use (default: claude-3-5-sonnet-20240620)')
    parser.add_argument('--format', '-f', 
                      choices=['markdown', 'html'], 
                      default='markdown', 
                      help='Output format (default: markdown)')
    parser.add_argument('--combine', '-c', 
                      action='store_true',
                      help='Combine multiple pages into a single lesson (for directory input)')
    parser.add_argument('--preview', '-p', 
                      action='store_true',
                      help='Show a preview of the generated lesson')
    parser.add_argument('--check', 
                      action='store_true',
                      help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    # Print banner
    console.print(Panel.fit(
        "[bold blue]EduParse[/bold blue] - [cyan]Convert scanned textbook pages to structured lessons[/cyan]",
        border_style="green"
    ))
    
    # Check for dependencies flag
    if args.check:
        console.print("[bold]Checking dependencies...[/bold]")
        check_dependencies()
        return
    
    # Validate dependencies
    if not check_dependencies():
        console.print("[bold red]Please install missing dependencies before continuing.[/bold red]")
        return
    
    # Validate Claude API key
    if not validate_api_key(args.api_key):
        return
    
    # Process input
    if os.path.isfile(args.input):
        console.print(f"[bold]Processing single file: {args.input}[/bold]")
        output_path = process_textbook_page(
            args.input, args.output, args.api_key, args.model, args.format
        )
        
        if output_path and args.preview:
            display_preview(output_path)
            
    else:  # Directory
        if not os.path.isdir(args.input):
            console.print(f"[bold red]Input path not found: {args.input}[/bold red]")
            return
            
        console.print(f"[bold]Processing directory: {args.input}[/bold]")
        processed_files = process_multiple_pages(
            args.input, args.output, args.api_key, args.model, args.format, args.combine
        )
        
        if processed_files and args.preview:
            display_preview(processed_files[-1])
    
    console.print("\n[bold green]Processing complete![/bold green]")

if __name__ == "__main__":
    main() 