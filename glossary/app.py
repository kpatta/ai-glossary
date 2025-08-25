import markdown
from flask import Flask, render_template_string, request
import os
import re

app = Flask(__name__)

MARKDOWN_PATH = "./AI-Comprehensive-Glossary.md"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Comprehensive Glossary</title>
    <style>
        html {
            scroll-behavior: smooth;
        }
        :root {
            --bg-gradient-dark: linear-gradient(120deg, #232526 0%, #414345 100%);
            --bg-gradient-light: linear-gradient(120deg, #f8fafc 0%, #e0e6ed 100%);
            --text-dark: #e0e6ed;
            --text-light: #232526;
            --search-bg-dark: #2c2f36;
            --search-bg-light: #f0f4fa;
            --accent: #00c3ff;
            --section-bg-dark: rgba(44,47,54,0.7);
            --section-bg-light: #f8fafc;
            --highlight-dark: linear-gradient(90deg, #00c3ff 0%, #ffffcc 100%);
            --highlight-light: linear-gradient(90deg, #00c3ff 0%, #e0e6ed 100%);
        }
        body {
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            margin: 2em;
            background: var(--bg-gradient-dark);
            color: var(--text-dark);
            transition: background 0.3s, color 0.3s;
        }
        body.light-mode {
            background: var(--bg-gradient-light);
            color: var(--text-light);
        }
        #search {
            width: 320px;
            padding: 10px;
            margin-bottom: 2em;
            border-radius: 8px;
            border: none;
            background: var(--search-bg-dark);
            color: var(--text-dark);
            font-size: 1.1em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: background 0.3s, color 0.3s;
        }
        body.light-mode #search {
            background: var(--search-bg-light);
            color: var(--text-light);
        }
        #search:focus {
            outline: 2px solid var(--accent);
        }
        .section {
            margin-bottom: 2em;
            padding: 1.2em 1em;
            border-radius: 10px;
            background: var(--section-bg-dark);
            box-shadow: 0 1px 6px rgba(0,0,0,0.07);
            transition: background 0.3s;
        }
        body.light-mode .section {
            background: var(--section-bg-light);
        }
        h2, h3 {
            margin-top: 2em;
            color: var(--accent);
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        h1 {
            color: var(--accent);
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 0.5em;
            letter-spacing: 1px;
        }
        .highlight {
            background: var(--highlight-dark);
            color: #232526;
            border-radius: 3px;
            padding: 0 2px;
            transition: background 0.3s, color 0.3s;
        }
        body.light-mode .highlight {
            background: var(--highlight-light);
            color: var(--text-light);
        }
        ::selection {
            background: var(--accent);
            color: #232526;
        }
        a {
            color: var(--accent);
            text-decoration: underline;
        }
        .toggle-btn {
            position: fixed;
            top: 24px;
            right: 50px;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 20px;
            padding: 8px 18px;
            font-size: 1em;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            z-index: 100;
            transition: background 0.3s, color 0.3s;
        }
        /* Alphabet navigation - Smart hover system */
        .alphabet-nav {
            position: fixed;
            top: 50%;
            right: -38px;
            transform: translateY(-50%);
            background: var(--section-bg-dark);
            padding: 0.8em 0.4em;
            border-radius: 8px 0 0 8px;
            box-shadow: -3px 0 10px rgba(0,0,0,0.3);
            z-index: 9999;
            opacity: 0.9;
            min-height: 300px;
            border: 2px solid var(--accent);
            transition: right 0.3s ease, opacity 0.3s ease;
        }
        .alphabet-nav:hover {
            right: 0;
            opacity: 1;
        }
        body.light-mode .alphabet-nav {
            background: var(--section-bg-light);
            box-shadow: -3px 0 10px rgba(0,0,0,0.2);
        }
        .alphabet-nav h3 {
            margin: 0 0 0.5em 0;
            color: var(--accent);
            font-size: 0.7em;
            text-align: center;
            padding: 0;
            writing-mode: horizontal-tb;
        }
        .alphabet-letters {
            display: flex;
            flex-direction: column;
            gap: 2px;
            align-items: center;
        }
        .alphabet-letter {
            display: block;
            width: 22px;
            height: 22px;
            line-height: 22px;
            text-align: center;
            background: var(--accent);
            color: #fff;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.7em;
            transition: all 0.2s;
            cursor: pointer;
            margin: 0;
        }
        body.light-mode .alphabet-letter {
            background: var(--accent);
            color: #fff;
        }
        .alphabet-letter:hover {
            background: #ff6b35;
            color: #fff;
            transform: scale(1.1);
        }
        .alphabet-letter.active {
            background: #ff6b35;
            color: #fff;
            transform: scale(1.1);
        }
        .alphabet-letter.disabled {
            opacity: 0.4;
            background: #666;
            cursor: not-allowed;
        }
        .alphabet-letter.disabled:hover {
            background: #666;
            color: #fff;
            transform: none;
        }
        /* Special styling for ALL button */
        .alphabet-letter.all-btn {
            background: var(--accent);
            color: #fff;
            font-weight: 700;
            font-size: 0.6em;
        }
        .alphabet-letter.all-btn:hover {
            background: var(--accent-hover);
            transform: scale(1.1);
        }
        .alphabet-letter.all-btn.active {
            background: var(--accent-hover);
            color: #fff;
        }
        /* Separator styling */
        .alphabet-separator {
            color: var(--accent);
            text-align: center;
            font-size: 0.8em;
            margin: 2px 0;
            pointer-events: none;
        }
        /* Alphabet nav tab indicator */
        .alphabet-nav::before {
            content: "A-Z";
            position: absolute;
            left: -18px;
            top: 50%;
            transform: translateY(-50%) rotate(-90deg);
            background: var(--accent);
            color: #fff;
            padding: 3px 6px;
            font-size: 0.6em;
            font-weight: 700;
            border-radius: 3px;
            white-space: nowrap;
            pointer-events: none;
            box-shadow: -2px 0 5px rgba(0,0,0,0.3);
        }
        /* Styles for anchor links and table of contents */
        h2:target {
            background: rgba(0, 195, 255, 0.1);
            padding: 0.5em;
            border-radius: 8px;
            transition: background 0.3s ease;
        }
        /* Ensure proper spacing for anchored sections */
        h2[id], h3[id] {
            scroll-margin-top: 2em;
        }
        h3:target {
            background: rgba(0, 195, 255, 0.1);
            padding: 0.3em;
            border-radius: 6px;
            transition: background 0.3s ease;
        }
    </style>
    <script>
        function searchGlossary() {
            var input = document.getElementById('search').value.toLowerCase();
            var words = input.split(' ').filter(word => word.length > 0);
            var sections = document.getElementsByClassName('section');
            var anyVisible = false;
            
            // Clear alphabet filter when searching
            if (input.length > 0) {
                var allLetters = document.querySelectorAll('.alphabet-letter');
                allLetters.forEach(function(btn) {
                    btn.classList.remove('active');
                });
            }
            
            for (var i = 0; i < sections.length; i++) {
                removeHighlights(sections[i]);
                var text = sections[i].textContent.toLowerCase();
                var match = words.length === 0 || words.every(function(word) { 
                    return text.includes(word); 
                });
                sections[i].style.display = match ? '' : 'none';
                if (match && input) {
                    highlightAllMatches(sections[i], words);
                }
                if (match) anyVisible = true;
            }
            
            var glossaryWrapper = document.getElementById('glossary-wrapper');
            var noResultsDiv = document.getElementById('no-results');
            if (anyVisible) {
                glossaryWrapper.style.display = '';
                noResultsDiv.style.display = 'none';
            } else {
                glossaryWrapper.style.display = 'none';
                noResultsDiv.style.display = '';
            }
        }

        function removeHighlights(element) {
            // Find all highlight spans
            var highlights = element.querySelectorAll('span.highlight');
            
            // Convert to array to avoid live NodeList issues
            var highlightArray = Array.prototype.slice.call(highlights);
            
            highlightArray.forEach(function(span) {
                var parent = span.parentNode;
                if (parent) {
                    // Replace the span with its text content
                    var textNode = document.createTextNode(span.textContent);
                    parent.replaceChild(textNode, span);
                }
            });
            
            // Normalize the parent to merge adjacent text nodes
            element.normalize();
        }

        function highlightAllMatches(element, words) {
            var walk = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);
            var textNodes = [];
            var node;
            
            // Collect all text nodes first
            while (node = walk.nextNode()) {
                textNodes.push(node);
            }
            
            // Process each text node
            textNodes.forEach(function(textNode) {
                var text = textNode.nodeValue;
                var parent = textNode.parentNode;
                var fragment = document.createDocumentFragment();
                var lastIndex = 0;
                var hasMatches = false;
                
                // Find all matches for all words
                var matches = [];
                words.forEach(function(word) {
                    if (word.length > 0) {
                        var regex = new RegExp(word.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), 'gi');
                        var match;
                        while ((match = regex.exec(text)) !== null) {
                            matches.push({
                                start: match.index,
                                end: match.index + match[0].length,
                                text: match[0]
                            });
                        }
                    }
                });
                
                // Sort matches by start position
                matches.sort(function(a, b) { return a.start - b.start; });
                
                // Remove overlapping matches (keep first occurrence)
                var filteredMatches = [];
                matches.forEach(function(match) {
                    var isOverlapping = filteredMatches.some(function(existing) {
                        return (match.start < existing.end && match.end > existing.start);
                    });
                    if (!isOverlapping) {
                        filteredMatches.push(match);
                    }
                });
                
                // Create text fragments with highlights
                filteredMatches.forEach(function(match) {
                    // Add text before the match
                    if (lastIndex < match.start) {
                        fragment.appendChild(document.createTextNode(text.substring(lastIndex, match.start)));
                    }
                    
                    // Add highlighted match
                    var span = document.createElement('span');
                    span.className = 'highlight';
                    span.textContent = match.text;
                    fragment.appendChild(span);
                    
                    lastIndex = match.end;
                    hasMatches = true;
                });
                
                // Add remaining text after last match
                if (lastIndex < text.length) {
                    fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
                }
                
                // Replace the text node with the fragment only if there were matches
                if (hasMatches) {
                    parent.replaceChild(fragment, textNode);
                }
            });
        }

        function toggleMode() {
            document.body.classList.toggle('light-mode');
        }

        function initializeAlphabetNav() {
            // Get all terms from the glossary sections
            var sections = document.getElementsByClassName('section');
            var terms = [];
            var availableLetters = new Set();
            
            for (var i = 0; i < sections.length; i++) {
                // Find all h3 headings (individual terms) within each section
                var h3Elements = sections[i].getElementsByTagName('h3');
                for (var j = 0; j < h3Elements.length; j++) {
                    var termText = h3Elements[j].textContent.trim();
                    if (termText) {
                        var firstLetter = termText.charAt(0).toUpperCase();
                        if (firstLetter.match(/[A-Z]/)) {
                            terms.push({
                                element: h3Elements[j],
                                text: termText,
                                letter: firstLetter
                            });
                            availableLetters.add(firstLetter);
                        }
                    }
                }
            }
            
            // Create vertical alphabet navigation
            var alphabetNav = document.createElement('div');
            alphabetNav.className = 'alphabet-nav';
            alphabetNav.innerHTML = '<h3>Nav</h3><div class="alphabet-letters" id="alphabet-letters"></div>';
            
            var lettersContainer = alphabetNav.querySelector('.alphabet-letters');
            
            // Add "ALL" button first
            var allBtn = document.createElement('a');
            allBtn.className = 'alphabet-letter all-btn';
            allBtn.textContent = 'ALL';
            allBtn.setAttribute('data-letter', 'ALL');
            allBtn.title = 'Show all terms';
            allBtn.onclick = function(e) {
                e.preventDefault();
                navigateToLetter(this.getAttribute('data-letter'));
            };
            lettersContainer.appendChild(allBtn);
            
            // Add separator
            var separator = document.createElement('div');
            separator.className = 'alphabet-separator';
            separator.textContent = '─';
            lettersContainer.appendChild(separator);
            
            // Create letter buttons A-Z
            for (var i = 65; i <= 90; i++) {
                var letter = String.fromCharCode(i);
                var letterBtn = document.createElement('a');
                letterBtn.className = 'alphabet-letter';
                letterBtn.textContent = letter;
                letterBtn.setAttribute('data-letter', letter);
                letterBtn.title = 'Show terms starting with ' + letter;
                
                if (availableLetters.has(letter)) {
                    letterBtn.onclick = function(e) {
                        e.preventDefault();
                        navigateToLetter(this.getAttribute('data-letter'));
                    };
                } else {
                    letterBtn.classList.add('disabled');
                    letterBtn.title = 'No terms starting with ' + letter;
                }
                
                lettersContainer.appendChild(letterBtn);
            }
            
            // Append to body as fixed sidebar
            document.body.appendChild(alphabetNav);
        }
        
        function navigateToLetter(letter) {
            console.log('Filtering by letter:', letter);
            
            // Clear any existing search to avoid conflicts
            document.getElementById('search').value = '';
            
            // Remove active class from all letters
            var allLetters = document.querySelectorAll('.alphabet-letter');
            allLetters.forEach(function(btn) {
                btn.classList.remove('active');
            });
            
            // Add active class to clicked letter
            var clickedLetter = document.querySelector('[data-letter="' + letter + '"]');
            if (clickedLetter) {
                clickedLetter.classList.add('active');
            }
            
            // Get all sections and terms
            var sections = document.getElementsByClassName('section');
            var noResults = document.getElementById('no-results');
            var hasVisibleTerms = false;
            
            if (letter === 'ALL') {
                // Show all content
                for (var i = 0; i < sections.length; i++) {
                    sections[i].style.display = '';
                }
                hasVisibleTerms = true;
                noResults.style.display = 'none';
                console.log('Showing all terms');
            } else {
                // Filter content by selected letter
                for (var i = 0; i < sections.length; i++) {
                    var section = sections[i];
                    var h3Elements = section.getElementsByTagName('h3');
                    var sectionHasMatchingTerms = false;
                    
                    // Check each term in this section
                    for (var j = 0; j < h3Elements.length; j++) {
                        var termText = h3Elements[j].textContent.trim();
                        
                        if (termText && termText.charAt(0).toUpperCase() === letter) {
                            sectionHasMatchingTerms = true;
                            hasVisibleTerms = true;
                            console.log('Found matching term:', termText);
                            break;
                        }
                    }
                    
                    // Show/hide section based on whether it has matching terms
                    if (sectionHasMatchingTerms) {
                        section.style.display = '';
                        
                        // Within this section, hide non-matching terms
                        var allTerms = section.querySelectorAll('h3, p, ul, ol');
                        var currentTerm = null;
                        
                        for (var k = 0; k < allTerms.length; k++) {
                            var element = allTerms[k];
                            
                            if (element.tagName === 'H3') {
                                currentTerm = element;
                                var termText = element.textContent.trim();
                                
                                if (termText && termText.charAt(0).toUpperCase() === letter) {
                                    element.style.display = '';
                                } else {
                                    element.style.display = 'none';
                                }
                            } else if (currentTerm) {
                                // Show/hide content based on current term visibility
                                element.style.display = currentTerm.style.display;
                            }
                        }
                    } else {
                        section.style.display = 'none';
                    }
                }
                
                // Scroll to first visible term for smooth UX
                var firstVisibleTerm = null;
                for (var i = 0; i < sections.length && !firstVisibleTerm; i++) {
                    if (sections[i].style.display !== 'none') {
                        var h3Elements = sections[i].getElementsByTagName('h3');
                        for (var j = 0; j < h3Elements.length; j++) {
                            if (h3Elements[j].style.display !== 'none') {
                                firstVisibleTerm = h3Elements[j];
                                break;
                            }
                        }
                    }
                }
                
                if (firstVisibleTerm) {
                    setTimeout(function() {
                        firstVisibleTerm.scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start' 
                        });
                    }, 100);
                }
            }
            
            // Show/hide no results message
            noResults.style.display = hasVisibleTerms ? 'none' : 'block';
            
            console.log('Filter complete. Has visible terms:', hasVisibleTerms);
        }
        
        // Initialize alphabet navigation when page loads
        window.onload = function() {
            initializeAlphabetNav();
        };
    </script>
</head>
<body>
    <button class="toggle-btn" onclick="toggleMode()">Toggle Dark/Light Mode</button>
    <input type="text" id="search" onkeyup="searchGlossary()" placeholder="Search terms...">
    <div id="glossary-wrapper">
        {{ content|safe }}
    </div>
    <div id="no-results" style="display:none; text-align:center; color:#00c3ff; font-size:1.2em; margin-top:2em;">No results found.</div>
</body>
</html>
"""

def process_markdown(md_text):
    # Remove the first markdown heading '# AI Comprehensive Glossary' to avoid duplicate
    md_text = re.sub(r'^# AI Comprehensive Glossary\s*', '', md_text, flags=re.MULTILINE)
    
    # Remove duplicate sections by splitting on h2 headings and keeping only unique ones
    sections = re.split(r'^## ', md_text, flags=re.MULTILINE)
    unique_sections = []
    seen_headings = set()
    
    # Keep the content before first h2 (like table of contents)
    if sections:
        unique_sections.append(sections[0])
    
    # Process remaining sections and remove duplicates
    for section in sections[1:]:
        # Extract heading text (first line of the section)
        lines = section.split('\n', 1)
        if lines:
            heading = lines[0].strip()
            if heading not in seen_headings:
                seen_headings.add(heading)
                unique_sections.append(section)
    
    # Reconstruct the markdown with unique sections
    md_text = unique_sections[0]  # content before first h2
    for section in unique_sections[1:]:
        md_text += '\n## ' + section
    
    # Use markdown without toc extension to avoid conflicts
    html = markdown.markdown(md_text, extensions=['fenced_code', 'tables', 'attr_list'])
    
    # Manually add IDs to h2 headings to match the table of contents links exactly
    def add_heading_id(match):
        heading_text = match.group(1)
        
        # Fix the display text: convert &amp; back to & for proper display
        display_text = heading_text.replace('&amp;', '&')
        
        # Convert heading text to anchor ID format to match table of contents links exactly
        anchor_id = heading_text.lower()
        
        # First replace &amp; with &, then & with --, then spaces with hyphens
        # This matches how the table of contents links are generated
        anchor_id = anchor_id.replace('&amp;', '&')
        anchor_id = anchor_id.replace('&', '--')
        anchor_id = re.sub(r'\s+', '-', anchor_id)
        
        # Remove other special chars except hyphens and clean up multiple hyphens
        anchor_id = re.sub(r'[^\w-]', '', anchor_id)
        anchor_id = re.sub(r'-{3,}', '--', anchor_id).strip('-')  # Replace 3+ hyphens with 2
        
        return f'<h2 id="{anchor_id}">{display_text}</h2>'
    
    # Add IDs to h2 headings
    html = re.sub(r'<h2>([^<]*)</h2>', add_heading_id, html)
    
    # Add IDs to h3 headings (individual terms) for alphabet navigation
    def add_h3_id(match):
        heading_text = match.group(1)
        # Create simple anchor ID for terms
        anchor_id = heading_text.lower()
        anchor_id = re.sub(r'[^\w\s-]', '', anchor_id)
        anchor_id = re.sub(r'\s+', '-', anchor_id).strip('-')
        return f'<h3 id="{anchor_id}">{heading_text}</h3>'
    
    html = re.sub(r'<h3>([^<]*)</h3>', add_h3_id, html)
    
    # Wrap each <h2> and <h3> section in a div for search
    html = re.sub(r'(<h[23].*?>.*?</h[23]>)(.*?)(?=<h[23]|$)', 
                  lambda m: f'<div class="section">{m.group(1)}{m.group(2)}</div>', html, flags=re.DOTALL)
    
    return html

@app.route("/")
def index():
    if not os.path.exists(MARKDOWN_PATH):
        return "Glossary file not found."
    with open(MARKDOWN_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()
    html_content = process_markdown(md_text)
    return render_template_string(HTML_TEMPLATE, content=html_content)

if __name__ == "__main__":
    app.run(debug=True)