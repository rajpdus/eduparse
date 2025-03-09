"""Tests for EduParse core functionality."""
import os
import pytest
from pathlib import Path
import tempfile
import cv2
import numpy as np
from eduparse import (
    preprocess_image,
    extract_text_with_layout,
    detect_and_extract_images,
    generate_lesson_with_claude,
)

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a white background
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "Sample Text", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add a rectangle (simulating an image)
    cv2.rectangle(img, (200, 200), (400, 400), (0, 0, 0), -1)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        return tmp.name

@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.mark.unit
def test_preprocess_image(sample_image):
    """Test image preprocessing."""
    thresh, img = preprocess_image(sample_image)
    
    assert thresh is not None
    assert img is not None
    assert isinstance(thresh, np.ndarray)
    assert isinstance(img, np.ndarray)
    assert thresh.shape == img.shape[:2]

@pytest.mark.unit
def test_extract_text_with_layout(sample_image):
    """Test text extraction with layout preservation."""
    result = extract_text_with_layout(sample_image)
    
    assert isinstance(result, dict)
    assert 'raw_data' in result
    assert 'text_blocks' in result
    assert 'full_text' in result
    assert isinstance(result['full_text'], str)
    assert len(result['text_blocks']) > 0

@pytest.mark.unit
def test_detect_and_extract_images(sample_image, output_dir):
    """Test image detection and extraction."""
    extracted_images = detect_and_extract_images(sample_image, output_dir)
    
    assert isinstance(extracted_images, list)
    assert len(extracted_images) > 0
    
    for img_data in extracted_images:
        assert 'path' in img_data
        assert 'relative_path' in img_data
        assert 'position' in img_data
        assert 'normalized_position' in img_data
        assert os.path.exists(img_data['path'])

@pytest.mark.integration
def test_generate_lesson_with_claude(sample_image, output_dir, monkeypatch):
    """Test lesson generation with Claude."""
    # Mock the Anthropic client to avoid actual API calls
    class MockAnthropicClient:
        def messages(self):
            return self
            
        def create(self, *args, **kwargs):
            class MockResponse:
                content = [type('Content', (), {'text': 'Mock lesson content'})]
            return MockResponse()
    
    monkeypatch.setattr('anthropic.Anthropic', lambda *args, **kwargs: MockAnthropicClient())
    
    text_data = extract_text_with_layout(sample_image)
    extracted_images = detect_and_extract_images(sample_image, output_dir)
    
    lesson_content = generate_lesson_with_claude(
        text_data,
        extracted_images,
        'fake-api-key',
        'claude-3-sonnet-20240229'
    )
    
    assert isinstance(lesson_content, str)
    assert len(lesson_content) > 0

def test_cleanup(sample_image):
    """Clean up temporary test files."""
    if os.path.exists(sample_image):
        os.unlink(sample_image) 