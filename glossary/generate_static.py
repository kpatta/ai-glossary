#!/usr/bin/env python3
"""
Generate static HTML file for GitHub Pages deployment
"""

import markdown
import os
import re
from app import process_markdown, HTML_TEMPLATE

def generate_static_html():
    """Generate static HTML file for GitHub Pages"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the markdown file (try multiple possible locations)
    possible_paths = [
        os.path.join(script_dir, "AI-Comprehensive-Glossary.md"),
        os.path.join(script_dir, "exported-assets", "AI-Comprehensive-Glossary.md"),
        "AI-Comprehensive-Glossary.md",
        "exported-assets/AI-Comprehensive-Glossary.md"
    ]
    
    markdown_path = None
    for path in possible_paths:
        if os.path.exists(path):
            markdown_path = path
            break
    
    if not markdown_path:
        print("Error: AI-Comprehensive-Glossary.md not found!")
        print("Searched in the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    print(f"📄 Found markdown file: {markdown_path}")
    
    # Read the markdown content
    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    # Process the markdown using the same function as the Flask app
    html_content = process_markdown(md_text)
    
    # Generate the complete HTML using the template
    complete_html = HTML_TEMPLATE.replace("{{ content|safe }}", html_content)
    
    # Update the title to be more descriptive
    complete_html = complete_html.replace(
        "<title>AI Comprehensive Glossary</title>",
        "<title>Comprehensive AI, GenAI, Agentic AI & AIOps Glossary</title>"
    )
    
    # Write the static HTML file
    output_file = "./glossary/index.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(complete_html)
    
    print(f"✅ Static HTML generated successfully: {output_file}")
    print(f"📁 File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("\n🚀 GitHub Pages Setup Instructions:")
    print("1. Commit and push the index.html file to your repository")
    print("2. Go to your GitHub repository settings")
    print("3. Navigate to Pages section")
    print("4. Select 'Deploy from a branch'")
    print("5. Choose 'main' branch and '/ (root)' folder")
    print("6. Your glossary will be available at: https://[username].github.io/[repository-name]/")
    
    return True

if __name__ == "__main__":
    generate_static_html()
