import requests
import pdfplumber
import spacy
import sqlite3
import json
import time
import random
import io
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import warnings

# Configuration
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1'
]

# Create necessary directories
os.makedirs('data', exist_ok=True)

# Disable SSL verification warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Load standard spaCy model instead of the custom legal NER model
try:
    print("Loading spaCy NER model...")
    # Use a standard model instead of the custom legal one
    nlp = spacy.load("en_core_web_lg")
    print("Model loaded successfully")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

class LegalScraper:
    def __init__(self, rate_limit=2.5, verify_ssl=False):
        """Initialize the scraper with rate limiting to avoid getting blocked"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        })
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.verify_ssl = verify_ssl
        
    def _respect_rate_limit(self):
        """Enforce rate limiting between requests"""
        now = time.time()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last_request
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _rotate_user_agent(self):
        """Rotate the User-Agent to help avoid detection"""
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        
    def fetch_content(self, url, max_retries=3):
        """Robust content fetcher with retries and format detection"""
        self._respect_rate_limit()
        self._rotate_user_agent()
        
        try:
            print(f"Fetching: {url}")
            
            # Special handling for different domains
            domain = urlparse(url).netloc
            
            # For Indian Kanoon, try to use a different approach
            if 'indiankanoon.org' in domain:
                print("Using specialized headers for Indian Kanoon")
                # Add special headers known to work with Indian Kanoon
                headers = self.session.headers.copy()
                headers.update({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://indiankanoon.org/',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'same-origin',
                    'sec-fetch-user': '?1'
                })
                response = self.session.get(url, headers=headers, timeout=20, verify=self.verify_ssl)
            else:
                response = self.session.get(url, timeout=20, verify=self.verify_ssl)
                
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            # Handle PDF content
            if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
                print("Processing PDF document...")
                return self.process_pdf(response.content)
            else:
                print("Processing HTML content...")
                soup = BeautifulSoup(response.text, 'html.parser')
                # If it's a redirect page, follow the redirect
                meta_refresh = soup.find('meta', {'http-equiv': 'refresh'})
                if meta_refresh:
                    content = meta_refresh.get('content', '')
                    if 'url=' in content.lower():
                        redirect_url = content.split('url=')[1].strip()
                        redirect_url = urljoin(url, redirect_url)
                        print(f"Following redirect to: {redirect_url}")
                        return self.fetch_content(redirect_url, max_retries)
                return soup
                
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            if max_retries > 0:
                sleep_time = 2 ** (3 - max_retries) + random.uniform(1, 3)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                return self.fetch_content(url, max_retries-1)
            print(f"Failed to fetch {url} after multiple attempts")
            return None

    def process_pdf(self, content):
        """Extract text from PDF documents"""
        text = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text.append(extracted_text)
            return '\n'.join(text)
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None

    def extract_legal_entities(self, text):
        """Modified NER extraction to work with standard spaCy models"""
        if not text:
            return {"cases": [], "statutes": [], "judges": [], "jurisdictions": [], 
                    "dates": [], "organizations": [], "persons": []}
        
        # First clean the text if it has HTML remnants or navigation elements
        text = self._clean_text_for_entity_extraction(text)
        
        # Process text in chunks to avoid memory issues with large documents
        max_chunk_size = 25000  # Characters per chunk
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        entities = defaultdict(list)
        
        for chunk in chunks:
            try:
                doc = nlp(chunk)
                
                for ent in doc.ents:
                    # Map standard spaCy entity types to our legal categories
                    if ent.label_ == 'ORG':
                        # Skip navigation menu items and common web elements
                        if not self._is_navigation_element(ent.text):
                            entities['organizations'].append(ent.text)
                            # Courts are often recognized as organizations
                            if any(court in ent.text.lower() for court in ['court', 'tribunal', 'bench', 'judiciary']):
                                entities['jurisdictions'].append(ent.text)
                    elif ent.label_ == 'PERSON':
                        # Skip short names and likely not real persons
                        if len(ent.text.split()) > 1 and not self._is_navigation_element(ent.text):
                            # Check if this might be a malformed judge name like "MaheshwariJustice Hrishikesh"
                            # Typical valid judge names don't have "Justice" in the middle of the name
                            if "Justice" in ent.text and not ent.text.startswith("Justice"):
                                # Skip this entity as it's likely malformed
                                continue
                            
                            entities['persons'].append(ent.text)
                            # Try to identify judges
                            judge_indicators = ['justice', 'judge', 'chief justice', 'hon', 'j.', 'cji', 'honorable']
                            preceding_text = chunk[max(0, ent.start_char-50):ent.start_char].lower()
                            
                            # Better detection of judge names
                            if (any(title in preceding_text for title in judge_indicators) or
                                any(f"justice {name.lower()}" in chunk.lower() for name in ent.text.split())): 
                                entities['judges'].append(ent.text)
                    elif ent.label_ == 'DATE':
                        # Filter out navigation date elements and non-judgment dates
                        date_text = ent.text
                        if not self._is_navigation_element(date_text):
                            # Try to validate the date without using _is_valid_date
                            date_match = re.search(r'(19\d{2}|20\d{2})', date_text)
                            if date_match:
                                year = int(date_match.group(1))
                                if 1950 <= year <= datetime.now().year:
                                    entities['dates'].append(date_text)
                    elif ent.label_ == 'LAW':  # Some models might have this
                        entities['statutes'].append(ent.text)
                    elif ent.label_ == 'GPE':  # Geopolitical entities
                        if any(court in chunk[max(0, ent.start_char-30):ent.start_char+30].lower() 
                               for court in ['court', 'high court']):
                            entities['jurisdictions'].append(f"{ent.text} Court")
            except Exception as e:
                print(f"Error in NER processing: {str(e)}")
                # Continue with other chunks
        
        # Apply fallback methods for legal entities
        case_numbers = self.find_case_numbers(text)
        entities['cases'].extend(case_numbers)
        
        section_numbers = self.extract_section_numbers(text)
        entities['statutes'].extend(section_numbers)
        
        if not entities['dates']:
            dates = self.extract_dates(text)
            entities['dates'].extend(dates)
            
        # Extract potential case titles (e.g., "Smith v. Jones")
        case_titles = self.extract_case_titles(text)
        entities['cases'].extend(case_titles)
        
        # Remove duplicates and preserve order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
            
        # Remove navigation elements that might have been missed
        for key in entities:
            entities[key] = [e for e in entities[key] if not self._is_navigation_element(e)]
            
        return dict(entities)

    def _clean_text_for_entity_extraction(self, text):
        """Clean up text before entity extraction to remove navigation and irrelevant parts"""
        # Remove common navigation patterns
        patterns_to_remove = [
            r'Home\s+/\s+Top Stories\s+/', 
            r'More Consumer CasesBook ReviewsRounds UpsEvents',
            r'Supreme CourtHigh CourtHigh Court\s+All High Courts',
            r'Existing User AccountGift Premium\w+Subscribe Premium',
            r'Share this',
            r'Click here to read',
            r'Next Story',
            r'Tags[:\w\s]*$',
            r'Citation\s*:\s*\d{4}\s*LiveLaw',
        ]
        
        clean_text = text
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text)
            
        # Remove short paragraphs that look like navigation
        lines = clean_text.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip short navigation-like lines
            if len(line) < 5 or all(c.isupper() for c in line if c.isalpha()):
                continue
            filtered_lines.append(line)
            
        return '\n'.join(filtered_lines)
        
    def _is_navigation_element(self, text):
        """Check if text is likely a navigation element rather than a legal entity"""
        # Navigation elements often contain these patterns
        nav_indicators = [
            'home', 'subscribe', 'premium', 'next story', 'click here', 'share this',
            'user account', 'gift', 'tags', 'law firm', 'news update', 'top stories',
            'podcast', 'articles', 'tech &', 'law schools', 'job updates'
        ]
        
        text_lower = text.lower()
        
        # Check for navigation indicators
        if any(nav in text_lower for nav in nav_indicators):
            return True
            
        # Check if text is all uppercase (often menu items)
        if all(c.isupper() for c in text if c.isalpha()) and len(text) > 3:
            return True
            
        # Check if text appears to be a URL path component
        if '/' in text or text.startswith('#') or text.endswith('#'):
            return True
            
        return False

    def find_case_numbers(self, text):
        """Improved case number detection with prioritization"""
        # Primary case number patterns (direct case identifiers)
        primary_patterns = [
            r'SLP\(?C\)?\s*Nos?\.\s*\d+(?:-\d+)?(?:/\d{4})',
            r'Civil\s+Appeal\s+Nos?\.\s*\d+(?:-\d+)?\s+of\s+\d{4}',
            r'Criminal\s+Appeal\s+Nos?\.\s*\d+(?:-\d+)?\s+of\s+\d{4}',
            r'Writ\s+Petition\s*\([A-Z]+\)\s*No\.\s*\d+\s+of\s+\d{4}',
            r'Transfer\s+Petition\s*\([A-Z]+\)\s*No\.\s*\d+\s+of\s+\d{4}',
            r'Review\s+Petition\s*\([A-Z]+\)\s*No\.\s*\d+\s+of\s+\d{4}',
            r'Original\s+Suit\s+No\.\s*\d+\s+of\s+\d{4}'
        ]
        
        # Secondary case number patterns (less specific identifiers)
        secondary_patterns = [
            r'\b[A-Z]{2,4}\s?\d{1,4}\s?[/\-]\s?\d{4}\b',
            r'\bWP\(?C\)?\s?\d+\/\d{4}\b',
            r'\bC\.A\.\s?No\.\s?\d+\s?of\s?\d{4}\b'
        ]
        
        primary_matches = []
        secondary_matches = []
        
        # First look for primary case numbers
        for pattern in primary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            primary_matches.extend([match.strip() for match in matches])
            
        # Then look for secondary case numbers
        for pattern in secondary_patterns:
            matches = re.findall(pattern, text)
            secondary_matches.extend([match.strip() for match in matches])
        
        # Combine with priority given to primary matches
        all_matches = list(dict.fromkeys(primary_matches + secondary_matches))
        return all_matches
    
    def extract_case_titles(self, text):
        """Extract case titles in the format 'X vs Y' or 'X v. Y'"""
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+[Vv][sS]\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
            r'\b[A-Z][a-z]+\s+[Vv]\.\s+[A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[Vv][Ee][Rr][Ss][Uu][Ss]\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text))
        return list(set(matches))
        
    def extract_dates(self, text):
        """Extract dates from text"""
        patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b'
        ]
        
        matches = []
        for pattern in patterns:
            raw_matches = re.findall(pattern, text, re.IGNORECASE)
            # Clean up any trailing non-date text
            for match in raw_matches:
                # Extract just the date part using a more strict pattern
                date_only = re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', match, re.IGNORECASE)
                if date_only:
                    matches.append(date_only.group())
        
        # Filter out obviously wrong dates (like years before 1950 or after 2030)
        valid_matches = []
        for match in matches:
            year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', match)
            if year_match:
                year = int(year_match.group(1))
                if 1950 <= year <= 2030:
                    valid_matches.append(match)
            else:
                valid_matches.append(match)
                    
        return list(set(valid_matches))

    def extract_section_numbers(self, text):
        """Extract section numbers from various acts with improved accuracy"""
        valid_patterns = [
            r'[Ss]ection\s+\d+(?:[A-Z])?(?:\(\d+\))?(?:\(\w\))?\s+of\s+the\s+[A-Za-z\s]+Act',
            r'[Ss]ection\s+\d+(?:[A-Z])?(?:\(\d+\))?(?:\(\w\))?\s+of\s+[A-Za-z\s]+Act',
            r'[Ss]\.\s*\d+(?:[A-Z])?(?:\(\d+\))?(?:\(\w\))?\s+of\s+the\s+[A-Za-z\s]+Act'
        ]
        
        # More restricted pattern for standalone section references
        standalone_patterns = [
            r'[Ss]ection\s+\d+(?:[A-Z])?(?:\(\d+\))?(?:\(\w\))?',
            r'[Ss]\.\s*\d+(?:[A-Z])?(?:\(\d+\))?(?:\(\w\))?'
        ]
        
        # Patterns that are likely false positives (exclude these)
        exclude_patterns = [
            r'SLP\s*\([A-Z]\)\s*Nos?\.\s*\d+',
            r'Case\s+No\.\s*\d+',
            r'Appeal\s+No\.\s*\d+'
        ]
        
        matches = []
        # First, check for statute sections with act names
        for pattern in valid_patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        
        # Then, check for standalone sections, but with more scrutiny
        standalone_matches = []
        for pattern in standalone_patterns:
            potential_matches = re.findall(pattern, text)
            for match in potential_matches:
                # Check if this match overlaps with any exclude patterns
                is_excluded = False
                for ex_pattern in exclude_patterns:
                    context_start = max(0, text.find(match) - 20)
                    context_end = min(len(text), text.find(match) + len(match) + 20)
                    context = text[context_start:context_end]
                    if re.search(ex_pattern, context, re.IGNORECASE):
                        is_excluded = True
                        break
                
                if not is_excluded:
                    standalone_matches.append(match)
        
        # Prioritize matches with act names
        matches.extend(standalone_matches)
        return list(set(matches))

    def _is_valid_date(self, date_str):
        """Check if a date string is valid and not in the future"""
        if not date_str:
            return False
            
        # Try to parse the date string
        try:
            # Handle ISO format with timezone
            if 'T' in date_str and ('+' in date_str or 'Z' in date_str):
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', 
                           '%d %B, %Y', '%d %B %Y', '%B %d, %Y', '%B %d %Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched
                    return False
                    
            # Check if date is not in the future and after 1950
            now = datetime.now()
            return 1950 <= dt.year <= now.year and dt <= now
        except:
            return False

class LegalDataPipeline:
    def __init__(self, db_path='legal_data.db'):
        self.scraper = LegalScraper(verify_ssl=False)  # Disable SSL verification 
        self.db = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables with improved schema"""
        self.db.execute('''CREATE TABLE IF NOT EXISTS judgments
             (id TEXT PRIMARY KEY, 
              source TEXT, 
              title TEXT,
              court TEXT,
              judgment_date TEXT,
              content TEXT, 
              entities TEXT, 
              metadata TEXT,
              created_at TEXT)''')
        
        self.db.execute('''CREATE TABLE IF NOT EXISTS cases
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              judgment_id TEXT,
              case_number TEXT,
              FOREIGN KEY(judgment_id) REFERENCES judgments(id))''')
              
        self.db.execute('''CREATE TABLE IF NOT EXISTS statutes
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              judgment_id TEXT,
              statute_name TEXT,
              FOREIGN KEY(judgment_id) REFERENCES judgments(id))''')
              
        self.db.execute('''CREATE TABLE IF NOT EXISTS judges
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              judgment_id TEXT,
              judge_name TEXT,
              FOREIGN KEY(judgment_id) REFERENCES judgments(id))''')
        
        self.db.commit()

    def process_url(self, url, content=None):
        """Main processing pipeline"""
        if content is None:
            content = self.scraper.fetch_content(url)
            
        if not content:
            print(f"Failed to process {url}")
            return None
            
        # Process all URLs through the standard extraction pipeline
            
        if isinstance(content, BeautifulSoup):
            # Check if this is a category/listing page that should be skipped
            if self._is_listing_page(content, url):
                print(f"Skipping category/listing page: {url}")
                return None
                
            text, title = self._extract_from_html(content, url)
        else:
            text = content
            title = self._extract_title_from_text(text) or "Unnamed Judgment"
            
        if not text:
            print(f"No text content found at {url}")
            return None
            
        # Extract entities from the text
        entities = self.scraper.extract_legal_entities(text)
        
        # For LiveLaw, supplement with direct HTML extraction which is more reliable
        if 'livelaw.in' in url and isinstance(content, BeautifulSoup):
            self._extract_livelaw_specific_entities(content, entities)
            
        metadata = self._extract_metadata(url, content, title, entities)
        
        # Special case for LiveLaw URLs - make a direct targeted extraction attempt
        if 'livelaw.in' in url and isinstance(content, BeautifulSoup):
            # Direct extraction for specific metadata that might be missing
            self._direct_targeted_extraction(content, text, metadata)
            
        doc_id = self._generate_id(url)
        self._store_data(doc_id, url, text, title, entities, metadata)
        return entities
        
    def _direct_targeted_extraction(self, soup, text, metadata):
        """Directly target and extract specific metadata that might be missed by other methods"""
        # If we're missing primary_case_title or primary_citation, try direct targeted extraction
        print(f"Direct extraction - Current metadata: primary_case_title={metadata.get('primary_case_title', 'None')}, primary_citation={metadata.get('primary_citation', 'None')}")
        
        if not metadata.get('primary_case_title') or not metadata.get('primary_citation'):
            article_content = soup.find('div', {'class': ['entry-content', 'td-post-content']})
            if not article_content:
                return
            
            # Get the article text
            article_text = article_content.get_text()
            
            # Look for specific format at the end of LiveLaw articles
            # Try to find the exact Case: and Citation: lines
            case_line = None
            citation_line = None
            
            # First check for the standard footer format
            footer_pattern = re.search(r'Case\s*:(.+?)Coram\s*:(.+?)Citation\s*:(.+?)(?:Click here|$)', 
                                       article_text, re.DOTALL | re.IGNORECASE)
                                       
            if footer_pattern:
                case_line = footer_pattern.group(1).strip()
                coram_line = footer_pattern.group(2).strip()
                citation_line = footer_pattern.group(3).strip()
                
                print(f"Direct extraction - found case line: {case_line}")
                print(f"Direct extraction - found citation line: {citation_line}")
            else:
                # Try looking at paragraphs near the end
                paragraphs = article_content.find_all('p')
                for p in paragraphs[-10:]:  # Check last 10 paragraphs
                    p_text = p.get_text().strip()
                    if p_text.startswith('Case:') or 'Case:' in p_text:
                        case_line = re.sub(r'^.*?Case\s*:\s*', '', p_text, flags=re.IGNORECASE).strip()
                        print(f"Found case line in paragraph: {case_line}")
                    elif p_text.startswith('Citation:') or 'Citation:' in p_text:
                        citation_line = re.sub(r'^.*?Citation\s*:\s*', '', p_text, flags=re.IGNORECASE).strip()
                        print(f"Found citation line in paragraph: {citation_line}")
            
            # Extract case title if needed
            if not metadata.get('primary_case_title') and case_line:
                # Extract case title and number - typically in format "TITLE [CASE_NUMBER]"
                case_parts = re.match(r'(.*?)\s*\[(.*?)\]', case_line)
                if case_parts:
                    primary_title = case_parts.group(1).strip()
                    if primary_title:
                        print(f"Direct extraction of case title: {primary_title}")
                        metadata['primary_case_title'] = primary_title
                else:
                    # If no brackets, use the whole line as title
                    metadata['primary_case_title'] = case_line
            
            # Extract citation if needed
            if not metadata.get('primary_citation'):
                if citation_line:
                    print(f"Setting citation from line: {citation_line}")
                    metadata['primary_citation'] = citation_line
                else:
                    # Try a direct search for the LiveLaw citation format
                    citation_match = re.search(r'(LL\s+\d{4}\s+SC\s+\d+)', article_text, re.IGNORECASE)
                    if citation_match:
                        citation = citation_match.group(1).strip()
                        print(f"Direct extraction of citation: {citation}")
                        metadata['primary_citation'] = citation
            
            # Last search attempt for citation in the article
            if not metadata.get('primary_citation'):
                citation_match = re.search(r'Citation\s*:\s*(LL\s+\d{4}\s+\w+\s+\d+)', article_text, re.IGNORECASE | re.DOTALL)
                if citation_match:
                    citation = citation_match.group(1).strip()
                    print(f"Final regex extraction of citation: {citation}")
                    metadata['primary_citation'] = citation
            
            # Use only dynamically extracted values, no hardcoded fallbacks

    def _is_listing_page(self, soup, url):
        """Detect if a page is a category/listing page rather than an individual judgment"""
        # URLs that look like listing/category pages
        if any(x in url for x in ['/tags/', '/category/', '/high-court/', '/supreme-court/']):
            # Check if URL path ends with a listing segment
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            
            # If path ends with just the court name, it's likely a listing
            if len(path_parts) > 0 and path_parts[-1] in ['judgments', 'supreme-court', 
                'allahabad-high-court', 'bombay-high-court', 'delhi-high-court',
                'madras-high-court', 'calcutta-high-court', 'karnataka-high-court']:
                return True
            
            # Look for many article entries or a paginated list
            # LiveLaw articles could be in different containers
            articles = soup.find_all('article')
            if not articles:
                # Try alternative class names for articles/posts
                articles = soup.find_all(['div', 'li'], {'class': ['post', 'entry', 'article-item', 'td_module_16', 'td_module_wrap']})
            
            if len(articles) > 3:
                return True
                
            # Look for pagination elements
            pagination = soup.find('div', {'class': ['pagination', 'nav-links', 'page-nav', 'td-ss-main-content']})
            if pagination:
                return True
                
            # Look for "Latest News" or similar headers
            headers = soup.find_all(['h1', 'h2'])
            for header in headers:
                text = header.get_text().lower()
                if any(term in text for term in ['latest', 'news', 'updates', 'archives', 'listing']):
                    return True
        
        return False

    def _extract_from_html(self, soup, url):
        """Site-specific content extraction based on domain"""
        domain = urlparse(url).netloc
        title = soup.title.string if soup.title else ""
        
        # For LiveLaw, extract proper title from article header, not page title
        if 'livelaw.in' in domain:
            # First, remove all navigation, header, footer elements from the soup
            for nav_elem in soup.find_all(['nav', 'header', 'footer', 'aside']):
                nav_elem.decompose()
                
            for menu_elem in soup.find_all('div', class_=lambda c: c and any(x in c for x in ['menu', 'navigation', 'nav-', 'td-header', 'td-footer'])):
                menu_elem.decompose()
                
            # Find the main article tag
            article = soup.find('article')
            if article:
                # Try to extract a better title from the article header
                article_title = article.find('h1', {'class': 'entry-title'})
                if article_title:
                    title = article_title.get_text().strip()
                else:
                    # Try alternative title selectors
                    alt_title = soup.find('h1', {'class': ['tdb-title-text', 'td-page-title']})
                    if alt_title:
                        title = alt_title.get_text().strip()
                
                # Extract the actual judgment content - focus on the main article body
                article_content = article.find('div', {'class': ['entry-content', 'td-post-content']})
                if article_content:
                    # Clean up content by removing navigation, ads, etc.
                    # First remove any script, style, nav elements
                    for element in article_content.find_all(['script', 'style', 'nav', 'aside', 'footer']):
                        element.decompose()
                        
                    # Remove social sharing buttons and tags list
                    for element in article_content.find_all('div', {'class': ['td-post-sharing', 'td-post-source-tags']}):
                        element.decompose()
                        
                    # Remove author/date information elements
                    for element in article_content.find_all(['div', 'span'], {'class': ['author', 'date', 'byline', 'td-post-author-name']}):
                        element.decompose()
                    
                    # Extract only the paragraphs, headings, and blockquotes for better content
                    content_elements = article_content.find_all(['p', 'h2', 'h3', 'h4', 'blockquote'])
                    if content_elements:
                        # Filter out very short paragraphs that are likely not part of the judgment
                        clean_elements = [elem for elem in content_elements 
                                         if len(elem.get_text().strip()) > 15 or
                                         elem.name in ['h2', 'h3', 'h4']]
                        
                        if clean_elements:
                            clean_content = "\n\n".join(elem.get_text().strip() for elem in clean_elements)
                            # Remove any remaining navigation text
                            clean_content = self._clean_content(clean_content)
                            return clean_content, title
                    
                    # If we couldn't extract clean elements, use the full content but clean it
                    content_text = article_content.get_text()
                    clean_content = self._clean_content(content_text)
                    return clean_content, title
                
                # Extract judgment text - often in an embedded PDF viewer or specific div
                judgment_div = article.find('div', {'class': 'judgement-embed'})
                if judgment_div:
                    return judgment_div.get_text(), title
                
                # Fallback to the whole article, but clean it up
                # Remove navigation, social links, etc
                for element in article.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                    element.decompose()
                
                # Clean the text
                article_text = article.get_text()
                clean_content = self._clean_content(article_text)
                return clean_content, title
            else:
                # If no article tag, try to get content from td-post-content
                content_div = soup.find('div', {'class': ['td-post-content', 'tdb-block-inner']})
                if content_div:
                    # Also try to find a better title
                    post_title = soup.find('h1', {'class': ['entry-title', 'tdb-title-text']})
                    if post_title:
                        title = post_title.get_text().strip()
                        
                    # Clean up content by extracting only paragraphs and headings
                    clean_elements = content_div.find_all(['p', 'h2', 'h3', 'h4', 'blockquote'])
                    if clean_elements:
                        filtered_elements = [elem for elem in clean_elements if len(elem.get_text().strip()) > 15]
                        if filtered_elements:
                            clean_content = "\n\n".join(elem.get_text().strip() for elem in filtered_elements)
                            clean_content = self._clean_content(clean_content)
                            return clean_content, title
                    
                    # Fallback to whole div but clean it
                    content_text = content_div.get_text()
                    clean_content = self._clean_content(content_text)
                    return clean_content, title
                    
        # Indian Kanoon specific extraction
        elif 'indiankanoon.org' in domain:
            # Extract title from the header
            header = soup.find('header')
            if header:
                h1 = header.find('h1')
                if h1:
                    title = h1.get_text().strip()
            
            # Main judgment content
            main_content = soup.find('div', {'class': 'judgments'})
            if main_content:
                return main_content.get_text(), title
            
            # Alternative structure
            doc_content = soup.find('div', {'id': 'doc_content'})
            if doc_content:
                return doc_content.get_text(), title
                
        # Supreme Court of India extraction
        elif 'sci.gov.in' in domain or 'main.sci.gov.in' in domain:
            # Try to extract a better title
            case_title = soup.find('div', {'class': 'case_title'})
            if case_title:
                title = case_title.get_text().strip()
                
            # Main judgment content    
            judgment_div = soup.find('div', {'id': 'jud'})
            if judgment_div:
                return judgment_div.get_text(), title
                
            # Alternative structure
            main_content = soup.find('div', {'class': 'content'})
            if main_content:
                return main_content.get_text(), title
                
        # High Court sites often use 'article' or 'content' divs
        elif any(hc in domain for hc in ['highcourt', 'hcourt', 'court.gov.in']):
            # Try to get a better title from possible elements
            page_title = soup.find('h1', {'class': ['page-title', 'title', 'judgement-title']})
            if page_title:
                title = page_title.get_text().strip()
                
            # Try all possible content selectors for high courts
            for selector in [
                {'tag': 'div', 'attr': {'class': 'judgement-content'}},
                {'tag': 'div', 'attr': {'class': 'judgment-content'}},
                {'tag': 'div', 'attr': {'class': 'content'}},
                {'tag': 'div', 'attr': {'class': 'main-content'}},
                {'tag': 'div', 'attr': {'id': 'main-content'}},
                {'tag': 'article'},
                {'tag': 'div', 'attr': {'id': 'content'}},
                {'tag': 'div', 'attr': {'class': 'case-details'}}
            ]:
                content_div = soup.find(selector['tag'], selector.get('attr', {}))
                if content_div:
                    return content_div.get_text(), title
        
        # AIR Online specific extraction
        elif 'aironline.in' in domain:
            # Try to extract a better title
            judgment_title = soup.find('h1', {'class': 'judgment-title'}) 
            if judgment_title:
                title = judgment_title.get_text().strip()
                
            # Main judgment content    
            main_content = soup.find('div', {'class': 'judgement-text'})
            if main_content:
                return main_content.get_text(), title
                
            # Alternative class name
            judgment_content = soup.find('div', {'class': 'judgment-content'})
            if judgment_content:
                return judgment_content.get_text(), title
        
        # Bar and Bench extraction
        elif 'barandbench.com' in domain:
            # Try to extract a better title
            article_title = soup.find('h1', {'class': 'tdb-title-text'})
            if article_title:
                title = article_title.get_text().strip()
                
            # Main content usually in article tag    
            article = soup.find('article')
            if article:
                content_div = article.find('div', {'class': 'tdb-block-inner'})
                if content_div:
                    return content_div.get_text(), title
                return article.get_text(), title
                
        # Generic fallback - try common content patterns
        for content_id in ['content', 'main', 'article', 'judgment', 'fulltext', 'case-content', 'case-details']:
            content_div = soup.find('div', {'id': content_id}) or soup.find('div', {'class': content_id})
            if content_div:
                return content_div.get_text(), title
                
        # Last resort: extract the most text-heavy div
        divs = soup.find_all('div')
        if divs:
            largest_div = max(divs, key=lambda x: len(x.get_text()))
            return largest_div.get_text(), title
            
        # If all else fails, get all text
        print(f"Using fallback text extraction for {url}")
        return soup.get_text(), title

    def _clean_content(self, text):
        """Clean up text content by removing navigation, headers, and other non-judgment content"""
        # Remove common navigation patterns and metadata
        patterns_to_remove = [
            r'Existing User.*?Premium',
            r'Home\s*/\s*Top Stories\s*/.*?',
            r'Share this',
            r'Click here to read',
            r'Next Story',
            r'Tags.*?$',
            r'More.*?Consumer Cases.*?Law$',
            r'Supreme CourtHigh Court.*?All High Courts.*?High Court$',
            r'Law Schools.*?School Admission',
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{1,2}:\d{2}\s+[AP]M\s+IST',
            r'LivLaw.*?\d+',
            r'LIVELAW NEWS NETWORK \d+.*?IST',
            r'Also Read.*?Supreme Court',
            r'Case Title:.*$',
            r'Citation :.*$',
            r'Click Here To Read.*$',
            r'Headnotes.*$',
            r'Summary:.*$'
        ]
        
        clean_text = text
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL|re.IGNORECASE)
            
        # Split into lines and filter out navigational lines
        lines = clean_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty or very short lines
            if not line or len(line) < 3:
                continue
                
            # Skip navigation-like lines
            if (all(c.isupper() for c in line if c.isalpha() and len(line) > 3) or
                any(nav in line.lower() for nav in [
                    'subscribe', 'click here', 'next story', 'previous', 'home', 
                    'livelaw', 'news network', 'tags', 'read more', 'judgment',
                    'download', 'judgment', 'user account', 'premium'
                ])):
                continue
            
            # Skip bylines (author and date)
            if re.search(r'^\w+\s+\w+\s+\d+\s+\w+\s+\d{4}\s+\d+:\d+\s+[AP]M', line):
                continue
                
            filtered_lines.append(line)
            
        # Remove first line if it's a duplicate title
        if len(filtered_lines) > 1 and filtered_lines[0] == filtered_lines[1]:
            filtered_lines.pop(0)
            
        # Join filtered lines and clean up extra whitespace
        result = '\n'.join(filtered_lines)
        # Replace multiple newlines with just two
        result = re.sub(r'\n{3,}', '\n\n', result)
        # Replace multiple spaces with single space
        result = re.sub(r' {2,}', ' ', result)
        
        return result.strip()

    def _extract_title_from_text(self, text):
        """Extract a judgment title from text content"""
        lines = text.split('\n')
        for i, line in enumerate(lines[:20]):  # Look in first 20 lines
            if 'vs' in line.lower() or 'versus' in line.lower() or 'v.' in line.lower():
                return line.strip()
        return None

    def _extract_metadata(self, url, content, title, entities):
        """Extract document metadata with improved site-specific extraction"""
        metadata = {
            'source': url,
            'title': title,
            'extracted_date': datetime.now().isoformat()
        }
        
        # First extract from HTML structure which is most reliable
        if isinstance(content, BeautifulSoup):
            html_metadata = self._extract_metadata_from_html(content)
            metadata.update(html_metadata)
        
        # If we still don't have a court, try to infer from URL/domain
        if 'court' not in metadata:
            domain = urlparse(url).netloc
            
            # Check for Supreme Court indicators
            if any(sc in url for sc in ['supremecourt', 'supreme-court', 'sci.gov.in']) or 'supreme court' in title.lower():
                metadata['court'] = 'Supreme Court of India'
                
            # Check for High Court indicators
            elif any(hc in url for hc in ['highcourt', 'high-court', 'hcourt']) or 'high court' in title.lower():
                # Try to identify which high court
                for hc in ['delhi', 'bombay', 'calcutta', 'madras', 'allahabad', 
                          'karnataka', 'kerala', 'punjab', 'haryana', 'patna',
                          'gujarat', 'andhra-pradesh', 'telangana', 'orissa',
                          'chhattisgarh', 'rajasthan', 'himachal', 'sikkim']:
                    if hc in url.lower() or hc in domain.lower() or hc in title.lower():
                        if hc == 'punjab-haryana' or (hc == 'punjab' and 'haryana' in url):
                            metadata['court'] = 'Punjab and Haryana High Court'
                        elif hc == 'andhra-pradesh':
                            metadata['court'] = 'Andhra Pradesh High Court'
                        elif hc == 'himachal':
                            metadata['court'] = 'Himachal Pradesh High Court'
                        else:
                            metadata['court'] = f"{hc.title()} High Court"
                        break
                
                # If specific high court not identified
                if 'court' not in metadata:
                    # Try to extract from URL path
                    path_parts = url.split('/')
                    for part in path_parts:
                        if 'high-court' in part:
                            court_name = part.replace('-high-court', '').replace('-', ' ').title()
                            if court_name:
                                metadata['court'] = f"{court_name} High Court"
                                break
                    
                    # Last resort generic
                    if 'court' not in metadata:
                        metadata['court'] = 'High Court'
            
            # LiveLaw specific URL patterns
            elif 'livelaw.in' in domain:
                if '/supreme-court/' in url:
                    metadata['court'] = 'Supreme Court of India'
                elif '/high-court/' in url:
                    # Try to extract specific High Court from URL path
                    path_parts = url.split('/')
                    for part in path_parts:
                        if part not in ['high-court', 'all-high-courts']:
                            if '-high-court' in part:
                                hc_name = part.replace('-high-court', '').replace('-', ' ').title()
                                metadata['court'] = f"{hc_name} High Court"
                                break
                            elif part in ['delhi', 'bombay', 'calcutta', 'madras', 'allahabad', 
                                        'karnataka', 'kerala', 'punjab-haryana', 'gujarat']:
                                if part == 'punjab-haryana':
                                    metadata['court'] = 'Punjab and Haryana High Court'
                                else:
                                    metadata['court'] = f"{part.title()} High Court"
                                break
        
        # If we still don't have a judgment date, try to extract from entities or content
        if 'judgment_date' not in metadata and entities.get('dates'):
            # Get the most likely judgment date from entities
            valid_dates = []
            high_priority_dates = []
            
            # First clean up dates to remove any extraneous text
            cleaned_dates = []
            for date in entities['dates']:
                # Extract just the date part from potentially messy strings
                date_match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}|\d{4}\s*\|\s*\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})', date, re.IGNORECASE)
                if date_match:
                    clean_date = date_match.group(1).strip()
                    # For formats like "2022 | 19 April 2022Coram", extract just the date part
                    if '|' in clean_date:
                        clean_date = clean_date.split('|')[1].strip()
                    # Remove any trailing non-date text
                    clean_date = re.sub(r'(Coram.*|[^0-9A-Za-z,\s./-]+.*)', '', clean_date).strip()
                    cleaned_dates.append(clean_date)
                else:
                    cleaned_dates.append(date)
            
            for date in cleaned_dates:
                # Skip obviously incorrect phrases
                if re.search(r'the\s+year\s+\d{4}', date, re.IGNORECASE):
                    continue
                    
                # Validate date format and range
                try:
                    # Check if it's likely a valid judgment date (not just any date mentioned in text)
                    if re.search(r'(19[5-9]\d|20[0-2]\d)', date):
                        # Context analysis - check if date appears near judgment indicators
                        if isinstance(content, BeautifulSoup):
                            text = content.get_text()
                            date_pos = text.find(date)
                            if date_pos > 0:
                                context = text[max(0, date_pos-100):min(len(text), date_pos+100)]
                                
                                # High priority indicators that strongly suggest this is the judgment date
                                high_priority_indicators = [
                                    'decided on', 'dated', 'judgment dated', 'order dated',
                                    'delivered on', 'pronounced on', 'passed on', 'judgment date',
                                    'date of judgment', 'date of order', 'case details', 'citation'
                                ]
                                
                                # Check for case number near date which strongly indicates judgment date
                                has_case_near = re.search(r'([A-Z]{2,4}\s?\d+\s?[Oo][Ff]\s?\d{4}|[Cc][Rr][Aa]\s?\d+\s?[Oo][Ff]\s?\d{4})', context)
                                
                                if any(indicator in context.lower() for indicator in high_priority_indicators) or has_case_near:
                                    # This date is likely the actual judgment date
                                    high_priority_dates.append(date)
                                    continue  # Continue to check other dates too
                        
                        # No strong context, but still a potential date
                        valid_dates.append(date)
                except:
                    # Not a valid date format
                    continue

            # Prioritize dates: first try high priority dates, then standard valid dates
            combined_dates = high_priority_dates + valid_dates
            
            # If we found any dates, use the most appropriate one
            if combined_dates:
                # Try to parse dates to find most relevant one (prefer full dates over just years)
                current_year = datetime.now().year
                parsed_dates = []
                
                for date_str in combined_dates:
                    try:
                        # Try different date formats (prioritizing full dates)
                        for fmt in ['%d %B %Y', '%B %d, %Y', '%B %d %Y', '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y']:
                            try:
                                dt = datetime.strptime(date_str, fmt)
                                # Only accept dates between 1950 and current year
                                if 1950 <= dt.year <= current_year:
                                    # Give higher priority to full dates (day, month, year)
                                    priority = 1
                                    parsed_dates.append((dt, date_str, priority))
                                    break
                            except ValueError:
                                continue
                    except:
                        # If this date couldn't be parsed with standard formats, 
                        # it might be just a year or have a non-standard format
                        continue
                
                if parsed_dates:
                    # First prioritize by source (high priority dates first), then by format completeness, then by recency
                    parsed_dates.sort(key=lambda x: (-x[2], -x[0].year, -x[0].month if hasattr(x[0], 'month') else 0))
                    metadata['judgment_date'] = parsed_dates[0][1]
                elif high_priority_dates:
                    # If we couldn't parse any dates but have high priority dates, use the first one
                    metadata['judgment_date'] = high_priority_dates[0]
                elif valid_dates:
                    # Last resort: use first valid date if none could be parsed
                    metadata['judgment_date'] = valid_dates[0]
        
        # If we don't have judges from HTML, try to get from entities
        if 'judges' not in metadata and entities.get('judges'):
            metadata['judges'] = ', '.join(entities['judges'])
            
        # If we don't have case references from HTML, try to get from entities
        if 'case_number' not in metadata and entities.get('cases'):
            metadata['case_number'] = ', '.join(entities['cases'][:3])  # First 3 cases only
         
        # Make sure we don't confuse publication date with judgment date
        # For news sites, the default date is often the publication date
        if 'publication_date' in metadata and 'judgment_date' not in metadata:
            # Don't use publication date as judgment date
            pass
            
        return metadata

    def _extract_metadata_from_html(self, soup):
        """Extract metadata from HTML structure with site-specific rules"""
        metadata = {}
        
        # Extract metadata from soup based on common patterns
        
        # Look for standard metadata in meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            # Publication date in meta tags
            if tag.get('property') == 'article:published_time' or tag.get('name') == 'publication_date':
                pub_date = tag.get('content', '')
                # Validate the date
                if pub_date and self._is_valid_date(pub_date):
                    metadata['publication_date'] = pub_date
            
            # Author info (could be judge)
            if tag.get('property') == 'article:author' or tag.get('name') == 'author':
                metadata['author'] = tag.get('content', '')
        
        # Try specific LiveLaw patterns
        if soup.find('a', {'href': re.compile(r'livelaw\.in')}):
            # Publication date is often in the publish-date span or time elements
            date_elem = soup.find(['span', 'time'], {'class': ['publish-date', 'entry-date', 'updated']})
            if date_elem:
                pub_date = date_elem.get_text().strip()
                if self._is_valid_date(pub_date):
                    metadata['publication_date'] = pub_date
            
            # Also look for author byline which sometimes includes the date
            author_byline = soup.find('div', {'class': 'td-post-author-name'})
            if author_byline:
                byline_text = author_byline.get_text()
                
                # Extract author name
                author_match = re.search(r'([A-Za-z\s]+)', byline_text)
                if author_match:
                    metadata['author'] = author_match.group(1).strip()
                
                # Try to extract publication date from byline
                date_match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})', byline_text)
                if date_match and self._is_valid_date(date_match.group(1)):
                    metadata['publication_date'] = date_match.group(1)
            
            # For LiveLaw, we need to find the actual judgment date in the content
            # This is different from the publication date of the article
            article_content = soup.find('div', {'class': ['entry-content', 'td-post-content']})
            if article_content:
                # Get the full article text to extract metadata sections more reliably
                article_text = article_content.get_text()
                
                # Look for the case, coram, and citation sections near the end of the article
                # These sections typically follow each other at the end of LiveLaw articles
                # Use more flexible patterns to match line endings and handle multiple line endings
                case_match = re.search(r'Case\s*:\s*(.*?)(?:[\n\r]+|Citation:|Coram:|$)', article_text, re.IGNORECASE | re.DOTALL)
                coram_match = re.search(r'Coram\s*:\s*(.*?)(?:[\n\r]+|Citation:|Case:|$)', article_text, re.IGNORECASE | re.DOTALL)
                citation_match = re.search(r'Citation\s*:\s*(.*?)(?:[\n\r]+|Coram:|Case:|$)', article_text, re.IGNORECASE | re.DOTALL)
                
                # Look specifically for lines at the end of the article (often the most reliable)
                # Find paragraphs that contain these metadata sections
                for para in article_content.find_all('p'):
                    para_text = para.get_text().strip()
                    
                    # Case line - look for "Case:" followed by text
                    if para_text.startswith('Case:') or 'Case:' in para_text:
                        case_para_match = re.search(r'Case\s*:\s*(.*?)(?:$)', para_text, re.IGNORECASE)
                        if case_para_match:
                            case_match = case_para_match
                            print(f"Found case in paragraph: {case_para_match.group(1).strip()}")
                    
                    # Citation line - look for "Citation:" followed by text
                    if para_text.startswith('Citation:') or 'Citation:' in para_text:
                        citation_para_match = re.search(r'Citation\s*:\s*(.*?)(?:$)', para_text, re.IGNORECASE)
                        if citation_para_match:
                            citation_match = citation_para_match
                            print(f"Found citation in paragraph: {citation_para_match.group(1).strip()}")
                
                # Extract case information - this is the most important part
                if case_match:
                    case_line = case_match.group(1).strip()
                    # Extract case title and number - typically in format "TITLE [CASE_NUMBER]"
                    case_parts = re.match(r'(.*?)\s*\[(.*?)\]', case_line)
                    if case_parts:
                        # Store the case title and number separately
                        primary_title = case_parts.group(1).strip()
                        primary_number = case_parts.group(2).strip()
                        if primary_title:
                            print(f"Extracted primary case title: {primary_title}")
                            metadata['primary_case_title'] = primary_title
                        if primary_number:
                            print(f"Extracted primary case number: {primary_number}")
                            metadata['primary_case_number'] = primary_number
                    else:
                        # If no brackets, use the whole line as title
                        print(f"Using whole line as case title: {case_line}")
                        metadata['primary_case_title'] = case_line
                else:
                    print("No case match found in the article")
                
                # Extract judge information with improved handling
                if coram_match:
                    coram_line = coram_match.group(1).strip()
                    # Clean up common prefixes
                    coram_line = re.sub(r'^Justices\s+', '', coram_line, flags=re.IGNORECASE)
                    
                    # Split judges by commas and 'and'
                    # First split by 'and' to handle the last judge properly
                    parts = re.split(r'\s+and\s+', coram_line, flags=re.IGNORECASE)
                    judges = []
                    
                    # Process all parts before the last 'and'
                    if len(parts) > 1:
                        # All but the last part should be split by commas
                        for part in parts[:-1]:
                            for name in re.split(r'\s*,\s*', part):
                                if name.strip():
                                    judges.append(name.strip())
                        # Add the last part (after 'and') without comma splitting
                        if parts[-1].strip():
                            judges.append(parts[-1].strip())
                    else:
                        # If no 'and' was found, try comma splitting
                        for name in re.split(r'\s*,\s*', coram_line):
                            if name.strip():
                                judges.append(name.strip())
                    
                    # Clean up the judge names to remove any malformed entries
                    clean_judges = []
                    for judge in judges:
                        # Skip any judge name containing "Justice" if not at the beginning
                        if "Justice" in judge and not judge.lower().startswith("justice"):
                            continue
                        # Remove any "Justice" prefix that might have been kept
                        judge = re.sub(r'^Justice\s+', '', judge, flags=re.IGNORECASE)
                        if judge.strip():
                            clean_judges.append(judge.strip())
                    
                    if clean_judges:
                        # Use distinct list to avoid duplicates
                        metadata['judges'] = ", ".join(list(dict.fromkeys(clean_judges)))
                
                # Extract citation - try multiple approaches
                citation_found = False
                if citation_match:
                    citation = citation_match.group(1).strip()
                    if citation:
                        print(f"Extracted primary citation: {citation}")
                        metadata['primary_citation'] = citation
                        citation_found = True
                else:
                    print("No citation match found in the article")
                    # Debug output to see the text content
                    print("Last 200 characters of article text:")
                    print(article_text[-200:])
                
                # If no citation found via regex, try to find it in paragraphs near the end
                if not citation_found:
                    for para in reversed(article_content.find_all('p')[-15:]):  # Check last 15 paragraphs
                        para_text = para.get_text().strip()
                        if 'citation' in para_text.lower() or ('ll' in para_text and 'sc' in para_text.lower()):
                            # Look for LiveLaw citation format (LL YYYY SC XXX)
                            ll_citation_match = re.search(r'(LL\s+\d{4}\s+SC\s+\d+)', para_text, re.IGNORECASE)
                            if ll_citation_match:
                                citation = ll_citation_match.group(1).strip()
                                print(f"Found citation in paragraph: {citation}")
                                metadata['primary_citation'] = citation
                                citation_found = True
                                break
                
                # First look for specific date patterns in the content
                judgment_date = None
                
                # Look for citation sections that often mention the judgment date
                citation_section = None
                for p in article_content.find_all('p'):
                    text = p.get_text().lower()
                    if ('citation' in text or 'cited on' in text or 'decided on' in text or 
                        'judgment dated' in text or 'order dated' in text or 'slp' in text or
                        'civil appeal no.' in text):
                        citation_section = p.get_text()
                        break
                
                if citation_section:
                    # Try to extract year from citation
                    year_match = re.search(r'\(\s*(\d{4})\s*\)', citation_section)
                    if year_match:
                        judgment_year = year_match.group(1)
                        if 1950 <= int(judgment_year) <= datetime.now().year:
                            judgment_date = judgment_year
                    
                    # Try to extract full date from citation
                    date_match = re.search(r'(?:dated|on|of)\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})', citation_section)
                    if date_match:
                        judgment_date = date_match.group(1)
                
                # If we couldn't find a citation section, look for full case citation at the end
                if not judgment_date:
                    for p in article_content.find_all('p'):
                        if re.search(r'\[\d{4}\]', p.get_text()):
                            # Likely a judgment citation with year
                            year_match = re.search(r'\[\s*(\d{4})\s*\]', p.get_text())
                            if year_match:
                                judgment_year = year_match.group(1)
                                if 1950 <= int(judgment_year) <= datetime.now().year:
                                    judgment_date = judgment_year
                
                if judgment_date:
                    metadata['judgment_date'] = judgment_date
            
            # Check article title for court mention
            article_title = soup.find('h1', {'class': ['entry-title', 'tdb-title-text']})
            if article_title:
                title_text = article_title.get_text()
                # Check for court mentions in title
                for court in ['Supreme Court', 'High Court']:
                    if court in title_text:
                        if court == 'Supreme Court':
                            metadata['court'] = 'Supreme Court of India'
                        elif 'High Court' in title_text:
                            # Try to extract which High Court
                            hc_match = re.search(r'([A-Za-z]+)\s+High Court', title_text)
                            if hc_match:
                                hc_name = hc_match.group(1)
                                metadata['court'] = f"{hc_name} High Court"
                            else:
                                metadata['court'] = 'High Court'
                                
        # Indian Kanoon specific patterns
        elif soup.find('a', {'href': re.compile(r'indiankanoon\.org')}):
            # Date is often present in a specific format
            date_div = soup.find('div', string=re.compile(r'Decided On:'))
            if date_div:
                date_match = re.search(r'Decided On:\s*([\d\w\s,]+)', date_div.get_text())
                if date_match and self._is_valid_date(date_match.group(1)):
                    metadata['judgment_date'] = date_match.group(1).strip()
            
            # Court information is often in a specific div
            court_div = soup.find('div', string=re.compile(r'Court:'))
            if court_div:
                court_match = re.search(r'Court:\s*(.+)', court_div.get_text())
                if court_match:
                    metadata['court'] = court_match.group(1).strip()
            
            # Judge information
            judges_div = soup.find('div', string=re.compile(r'Judges:'))
            if judges_div:
                judges_match = re.search(r'Judges:\s*(.+)', judges_div.get_text())
                if judges_match:
                    metadata['judges'] = judges_match.group(1).strip()
        
        # General patterns for other sites
        
        # Try to find a date
        date_patterns = [
            {'tag': 'div', 'attr': {'class': 'judgment-date'}},
            {'tag': 'div', 'attr': {'class': 'date'}},
            {'tag': 'span', 'attr': {'class': 'date'}},
            {'tag': 'time', 'attr': {'class': 'entry-date'}}
        ]
        
        for pattern in date_patterns:
            date_elem = soup.find(pattern['tag'], pattern.get('attr', {}))
            if date_elem:
                date_text = date_elem.get_text().strip()
                # Clean up the date text
                date_text = re.sub(r'Posted\s+on:|Date:', '', date_text).strip()
                if self._is_valid_date(date_text):
                    metadata['judgment_date'] = date_text
                break
                
        # Try to find court name
        court_patterns = [
            {'tag': 'div', 'attr': {'class': 'court-name'}},
            {'tag': 'div', 'attr': {'class': 'court'}},
            {'tag': 'span', 'attr': {'class': 'court'}},
            {'tag': 'div', 'attr': {'class': 'tribunal-name'}}
        ]
        
        for pattern in court_patterns:
            court_elem = soup.find(pattern['tag'], pattern.get('attr', {}))
            if court_elem:
                metadata['court'] = court_elem.get_text().strip()
                break
                
        return metadata

    def _generate_id(self, url):
        """Generate unique document ID"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('.', '_')
        path_parts = [p for p in parsed_url.path.split('/') if p]
        
        if path_parts:
            # Use last meaningful parts of the path
            path_id = '_'.join(path_parts[-3:]) if len(path_parts) >= 3 else '_'.join(path_parts)
        else:
            # Fallback with timestamp if path is empty
            path_id = f"doc_{int(time.time())}"
            
        return f"{domain}_{path_id}"
        
    def _is_valid_date(self, date_str):
        """Check if a date string is valid and not in the future"""
        if not date_str:
            return False
            
        # Try to parse the date string
        try:
            # Handle ISO format with timezone
            if 'T' in date_str and ('+' in date_str or 'Z' in date_str):
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y', 
                           '%d %B, %Y', '%d %B %Y', '%B %d, %Y', '%B %d %Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched
                    return False
                    
            # Check if date is not in the future and after 1950
            now = datetime.now()
            return 1950 <= dt.year <= now.year and dt <= now
        except:
            return False

    def _store_data(self, doc_id, source, text, title, entities, metadata):
        """Store data in SQLite and JSON backup with normalized schema"""
        current_time = datetime.now().isoformat()
        court = metadata.get('court', '')
        judgment_date = metadata.get('judgment_date', '')
        
        # Use primary case title from metadata if available, otherwise use the extracted title
        db_title = metadata.get('primary_case_title', title)
        
        # Clean up judge names in metadata to avoid malformed entries
        if 'judges' in metadata:
            # Parse the judge names and filter out any malformed entries
            judge_entries = [j.strip() for j in metadata['judges'].split(',')]
            clean_judges = []
            for judge in judge_entries:
                # Skip if this appears to be a malformed concatenation
                if "Justice" in judge and not judge.lower().startswith("justice"):
                    continue
                # Remove any "Justice" prefix
                judge = re.sub(r'^Justice\s+', '', judge, flags=re.IGNORECASE)
                if judge.strip():
                    clean_judges.append(judge.strip())
            
            # Update metadata with clean judges
            if clean_judges:
                metadata['judges'] = ", ".join(list(dict.fromkeys(clean_judges)))

        # Handle case information
        # First check for explicitly extracted primary case number from metadata
        primary_case_number = metadata.get('primary_case_number', '')
        primary_case_title = metadata.get('primary_case_title', '')
        primary_citation = metadata.get('primary_citation', '')
        
        # If no primary case number in metadata, try to find one from entities
        if not primary_case_number and entities.get('cases'):
            # Process case information to find primary case vs cited cases
            case_info = self._prioritize_case_info(entities.get('cases', []), text)
            primary_case_number = case_info.get('primary_case', '')
            
            # Add cited cases if available
            cited_cases = case_info.get('cited_cases', [])
            if cited_cases:
                metadata['cited_cases'] = ', '.join(cited_cases)
        # If primary_case_number was found in metadata, we need to identify other cases as cited
        elif primary_case_number and entities.get('cases'):
            # Find cases that are not the primary case and mark them as cited
            cited_cases = []
            
            for case in entities.get('cases', []):
                # Skip if this case is the primary case number or title
                if (case != primary_case_number and 
                    case != primary_case_title and 
                    not primary_case_number in case and 
                    (not primary_case_title or not primary_case_title in case)):
                    cited_cases.append(case)
            
            if cited_cases:
                metadata['cited_cases'] = ', '.join(cited_cases)
        
        # Ensure key fields are in metadata for consistency and JSON output
        if primary_case_number:
            metadata['primary_case_number'] = primary_case_number
            # Also set case_number for backward compatibility
            metadata['case_number'] = primary_case_number
        
        if primary_case_title:
            metadata['primary_case_title'] = primary_case_title
        
        if primary_citation:
            metadata['primary_citation'] = primary_citation
        
        # Rely on dynamically extracted metadata without hardcoded fallbacks
                
        # Main judgment entry
        try:
            self.db.execute('''
                INSERT OR REPLACE INTO judgments 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id, source, db_title, court, judgment_date, text, 
                json.dumps(entities), json.dumps(metadata), current_time
            ))
            
            # Related entities in separate tables
            # Store the primary case number if available
            if primary_case_number:
                self.db.execute('''
                    INSERT INTO cases (judgment_id, case_number)
                    VALUES (?, ?)
                ''', (doc_id, primary_case_number))

            # Store other referenced cases
            cited_cases_list = []
            if metadata.get('cited_cases'):
                cited_cases_list = [c.strip() for c in metadata.get('cited_cases').split(',')]
                
            for case_ref in entities.get('cases', []):
                # Skip if this is the primary case or already in cited cases
                if case_ref != primary_case_number and case_ref not in cited_cases_list:
                    self.db.execute('''
                        INSERT INTO cases (judgment_id, case_number)
                        VALUES (?, ?)
                    ''', (doc_id, case_ref))
                    
            if 'statutes' in entities:
                for statute in entities['statutes']:
                    self.db.execute('''
                        INSERT INTO statutes (judgment_id, statute_name)
                        VALUES (?, ?)
                    ''', (doc_id, statute))
                    
            if 'judges' in entities:
                # Clean up the judge names before storing
                clean_judge_entities = []
                for judge in entities['judges']:
                    # Skip malformed judge names like "MaheshwariJustice Hrishikesh"
                    if "Justice" in judge and not judge.lower().startswith("justice"):
                        continue
                    # Remove any "Justice" prefix
                    judge = re.sub(r'^Justice\s+', '', judge, flags=re.IGNORECASE)
                    if judge.strip():
                        clean_judge_entities.append(judge.strip())
                
                # Use the clean list for DB storage
                for judge in list(dict.fromkeys(clean_judge_entities)):
                    self.db.execute('''
                        INSERT INTO judges (judgment_id, judge_name)
                        VALUES (?, ?)
                    ''', (doc_id, judge))
                    
            self.db.commit()
            print(f"Stored document: {doc_id}")
            
            # Determine the source directory based on domain and court
            source_dir = self._get_source_directory(source, court)
            
            # Create directory if it doesn't exist
            os.makedirs(f'data/{source_dir}', exist_ok=True)
            
            # JSON backup with source-specific directory
            with open(f'data/{source_dir}/{doc_id}.json', 'w') as f:
                json.dump({
                    'id': doc_id,
                    'source': source,
                    'title': title,
                    'court': court,
                    'judgment_date': judgment_date,
                    'content': text,
                    'entities': entities,
                    'metadata': metadata,
                    'created_at': current_time
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error storing data: {str(e)}")
            
    def _get_source_directory(self, source, court):
        """Determine the appropriate directory based on source URL and court name"""
        # Extract domain from source URL
        domain = urlparse(source).netloc.replace('www.', '')
        
        # LiveLaw has specific court sections
        if 'livelaw.in' in domain:
            if 'Supreme Court' in court:
                return 'livelaw/supreme_court'
            elif 'High Court' in court:
                # Extract the specific high court name
                hc_match = re.search(r'(\w+)\s+High Court', court)
                if hc_match:
                    hc_name = hc_match.group(1).lower()
                    return f'livelaw/{hc_name}_hc'
                else:
                    return 'livelaw/high_courts'
            else:
                return 'livelaw/other'
                
        # Indian Kanoon
        elif 'indiankanoon.org' in domain:
            return 'indiankanoon'
            
        # Supreme Court of India official website
        elif 'sci.gov.in' in domain or 'main.sci.gov.in' in domain:
            return 'sci'
            
        # High Court websites
        elif any(hc in domain for hc in ['hcourt', 'highcourt', 'court.gov.in']):
            # Try to extract the specific high court from domain
            for hc in ['delhi', 'bombay', 'calcutta', 'madras', 'allahabad', 
                        'karnataka', 'kerala', 'punjab', 'haryana', 'patna',
                        'gujarat', 'telangana', 'orissa', 'rajasthan']:
                if hc in domain:
                    return f'{hc}_hc'
            return 'high_courts/other'
            
        # Default based on domain
        else:
            return domain.replace('.', '_')

    def _prioritize_case_info(self, cases, text=None):
        """Identify primary case number and cited cases from a list of case references"""
        if not cases:
            return {'primary_case': '', 'cited_cases': []}
            
        # Patterns for primary case numbers (direct identifiers)
        primary_patterns = [
            r'SLP\s*\([A-Z]\)\s*Nos?\.\s*\d+(?:-\d+)?[/\s]+\d{4}',
            r'Civil\s+Appeal\s+Nos?\.\s*\d+(?:-\d+)?\s+of\s+\d{4}',
            r'Criminal\s+Appeal\s+Nos?\.\s*\d+(?:-\d+)?\s+of\s+\d{4}',
            r'Writ\s+Petition\s*\([A-Z]+\)\s*No\.\s*\d+\s+of\s+\d{4}',
            r'WPC\s+\d+\s+OF\s+\d{4}'  # Add specific pattern for WPC format
        ]
        
        # Look for primary case number first
        primary_case = ''
        cited_cases = []
        
        # First check for case details section in text if provided
        if text:
            # Look for Case: line which indicates the primary case
            case_match = re.search(r'Case\s*:\s*(.*?)(?:\n|\r\n|$)', text, re.IGNORECASE)
            if case_match:
                case_line = case_match.group(1).strip()
                # Extract case number from square brackets if present
                case_number_match = re.search(r'\[(.*?)\]', case_line)
                if case_number_match:
                    primary_case = case_number_match.group(1).strip()
                    # Return early as this is the most reliable source
                    remaining_cases = [c for c in cases if c != primary_case and c != case_line.strip()]
                    return {'primary_case': primary_case, 'cited_cases': remaining_cases}
        
        # If no Case: line found, check for specific primary case number patterns
        for case in cases:
            for pattern in primary_patterns:
                if re.search(pattern, case, re.IGNORECASE):
                    primary_case = case
                    break
            if primary_case:
                break
                
        # If no primary case found, look for any case number format
        if not primary_case and cases:
            for case in cases:
                # Check if case looks like a case number rather than a case title
                if re.search(r'\d+\s*(?:of|\/)\s*\d{4}', case) or re.search(r'[A-Z]{2,4}\s?\d+[/\-]\d{4}', case):
                    primary_case = case
                    break
        
        # If still no primary case found, use the first case as primary
        if not primary_case and cases:
            primary_case = cases[0]
            
        # Collect cited cases (all cases except primary)
        cited_cases = [case for case in cases if case != primary_case]
            
        return {'primary_case': primary_case, 'cited_cases': cited_cases}
        
    def search_judgments(self, query, limit=10):
        """Search for judgments in the database"""
        cursor = self.db.cursor()
        
        try:
            # Full-text search would be better, but this is a simple implementation
            cursor.execute('''
                SELECT id, title, court, judgment_date, source
                FROM judgments
                WHERE content LIKE ? OR title LIKE ?
                ORDER BY judgment_date DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))
            
            results = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'title': row[1],
                    'court': row[2],
                    'date': row[3],
                    'source': row[4]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
            
    def _extract_livelaw_specific_entities(self, soup, entities):
        """Extract entities directly from LiveLaw's HTML structure"""
        # LiveLaw often has judge names in the article content
        article_content = soup.find('div', {'class': ['entry-content', 'td-post-content']})
        if not article_content:
            return
            
        # First try to extract from metadata sections at the end of the article
        article_text = article_content.get_text()
        paras = article_content.find_all('p')
        
        # Look for the formatted case, coram, and citation sections at the end
        coram_match = re.search(r'Coram\s*:\s*(.*?)(?:\n|\r\n|$)', article_text, re.IGNORECASE)
        case_match = re.search(r'Case\s*:\s*(.*?)(?:\n|\r\n|$)', article_text, re.IGNORECASE)
        citation_match = re.search(r'Citation\s*:\s*(.*?)(?:\n|\r\n|$)', article_text, re.IGNORECASE)
        
        # Extract case information from the structured Case: section
        if case_match:
            case_line = case_match.group(1).strip()
            
            # Extract case title and number - typically in format "TITLE [CASE_NUMBER]"
            case_parts = re.match(r'(.*?)\s*\[(.*?)\]', case_line)
            
            if case_parts:
                # Get the case title (part before brackets)
                case_title = case_parts.group(1).strip()
                case_number = case_parts.group(2).strip()
                
                # Add both the full case line, case title and case number to entities
                if case_line and case_line not in entities['cases']:
                    entities['cases'].append(case_line)
                
                if case_title and case_title not in entities['cases']:
                    entities['cases'].append(case_title)
                
                if case_number and case_number not in entities['cases']:
                    entities['cases'].append(case_number)
            else:
                # If no brackets, just add the whole line
                if case_line and case_line not in entities['cases']:
                    entities['cases'].append(case_line)
        
        # Extract judges from Coram section with improved handling
        if coram_match:
            coram_line = coram_match.group(1).strip()
            # Clean up common prefixes
            coram_line = re.sub(r'^Justices\s+', '', coram_line, flags=re.IGNORECASE)
            
            # Split judges by commas and 'and'
            # First split by 'and' to handle the last judge properly
            parts = re.split(r'\s+and\s+', coram_line, flags=re.IGNORECASE)
            judges = []
            
            # Process all parts before the last 'and'
            if len(parts) > 1:
                # All but the last part should be split by commas
                for part in parts[:-1]:
                    for name in re.split(r'\s*,\s*', part):
                        if name.strip():
                            judges.append(name.strip())
                # Add the last part (after 'and') without comma splitting
                if parts[-1].strip():
                    judges.append(parts[-1].strip())
            else:
                # If no 'and' was found, try comma splitting
                for name in re.split(r'\s*,\s*', coram_line):
                    if name.strip():
                        judges.append(name.strip())
            
            # Clean up the judge names and completely replace the existing judges list
            if judges:
                # Filter out malformed judges and remove duplicates
                clean_judges = []
                for judge in judges:
                    # Skip any judge name containing "Justice" if not at the beginning
                    if "Justice" in judge and not judge.lower().startswith("justice"):
                        continue
                    # Remove any "Justice" prefix that might have been kept
                    judge = re.sub(r'^Justice\s+', '', judge, flags=re.IGNORECASE)
                    if judge.strip():
                        clean_judges.append(judge.strip())
                
                if clean_judges:
                    # Completely replace any existing judges with our clean list
                    entities['judges'] = list(dict.fromkeys(clean_judges))
                    return
        
        # If no judges were extracted from the Coram section, try other methods
        judge_indicators = ['bench:', 'coram:', 'before:', 'justice', 'chief justice']
        
        # First try to find the bench/coram paragraph that lists judges
        judge_para = None
        for para in paras:
            text = para.get_text().lower()
            if any(indicator in text for indicator in judge_indicators):
                judge_para = para
                break
                
        # Extract judges from bench/coram paragraph
        if judge_para:
            # Extract the full paragraph text
            judge_text = judge_para.get_text()
            
            # Extract names that follow common judge prefixes
            judge_names = []
            for prefix in ['Justice', 'Hon\'ble', 'J.']:
                matches = re.finditer(fr"{prefix}\s+([A-Z][a-z]+\s+(?:[A-Z][a-z]*\s*)+)", judge_text)
                for match in matches:
                    judge_name = match.group(1).strip()
                    if judge_name and len(judge_name) > 3:
                        judge_names.append(judge_name)
            
            # For "bench:" or "coram:" patterns, extract all names
            if 'bench:' in judge_text.lower() or 'coram:' in judge_text.lower():
                judge_part = re.sub(r'^.*?(bench:|coram:)', '', judge_text, flags=re.IGNORECASE).strip()
                # Split judges by common separators
                for name in re.split(r',\s+|;\s+|and\s+', judge_part):
                    if name and any(char.isupper() for char in name):
                        # Clean up the name
                        clean_name = re.sub(r'^\s*(?:justice|j\.|hon\'ble)\s+', '', name, flags=re.IGNORECASE).strip()
                        if clean_name and len(clean_name) > 3:
                            judge_names.append(clean_name.title())
            
            # Only replace if we found judges this way
            if judge_names:
                entities['judges'] = list(dict.fromkeys(judge_names))  # Remove duplicates
        
        # If we didn't find judges yet, look in the case info section at the bottom
        if not entities.get('judges') or len(entities.get('judges', [])) == 0:
            # Look for lines like "Coram: Justice MR Shah and BV Nagarathna"
            for para in paras[-10:]:  # Check the last 10 paragraphs which often have case details
                text = para.get_text()
                if 'coram:' in text.lower() or 'bench:' in text.lower() or 'justice' in text.lower():
                    # Extract judge names
                    judge_names = []
                    judge_matches = re.findall(r'(?:Justice|J\.)\s+([A-Z][a-z]*\s+[A-Z][a-z]*|[A-Z]+\s+[A-Za-z]+)', text)
                    for judge in judge_matches:
                        if judge and judge.strip():
                            judge_names.append(judge.strip())
                    
                    # Look for "and" patterns for multiple judges
                    and_pattern = re.search(r'(?:Justice|J\.)\s+([A-Z][a-z]*\s+[A-Z][a-z]*)\s+and\s+([A-Z][a-z]*\s+[A-Z][a-z]*|[A-Z]+\s+[A-Za-z]+)', text)
                    if and_pattern:
                        first_judge = and_pattern.group(1).strip()
                        second_judge = and_pattern.group(2).strip()
                        if first_judge:
                            judge_names.append(first_judge)
                        if second_judge:
                            judge_names.append(second_judge)
                    
                    # Only update if we found judges
                    if judge_names:
                        entities['judges'] = list(dict.fromkeys(judge_names))  # Remove duplicates
                        break
                
        # If we still haven't found a case, look in paragraphs
        if not entities['cases']:
            case_indicators = ['case no.', 'petition no.', 'civil appeal', 'criminal appeal', 'writ petition', 'vs', 'versus']
            for para in paras:
                text = para.get_text().lower()
                if any(indicator in text for indicator in case_indicators):
                    # Common case number patterns
                    case_patterns = [
                        r'([A-Z]{2,4}\s?\d{1,4}\s?[/\-]\s?\d{4})',
                        r'(Civil\s?Appeal\s?No\.?\s?\d+\s?of\s?\d{4})',
                        r'(Criminal\s?Appeal\s?No\.?\s?\d+\s?of\s?\d{4})',
                        r'(Writ\s?Petition\s?\([A-Z]\)\s?No\.\s?\d+\s?of\s?\d{4})',
                        r'(SLP\s?\([A-Z]\)\s?No\.\s?\d+\s?of\s?\d{4})',
                        r'(WPC\s+\d+\s+OF\s+\d{4})'
                    ]
                    
                    for pattern in case_patterns:
                        matches = re.finditer(pattern, para.get_text(), re.IGNORECASE)
                        for match in matches:
                            case_num = match.group(1).strip()
                            if case_num and case_num not in entities['cases']:
                                entities['cases'].append(case_num)

class IndianLegalCrawler:
    """A specialized crawler for Indian legal websites"""
    
    def __init__(self, pipeline, max_pages=50):
        self.pipeline = pipeline
        self.scraper = pipeline.scraper
        self.max_pages = max_pages
        self.visited = set()
        
    def crawl_indiankanoon(self, start_url, limit=20):
        """Crawl Indian Kanoon judgments using search API instead of direct browsing"""
        print(f"Starting Indian Kanoon crawl using search approach")
        pages_processed = 0
        judgment_links = []
        
        # Indian Kanoon blocks direct access to browse pages
        # Instead we'll use their search functionality with proper user-agent and delays
        
        # Extract court name from the URL
        court_match = re.search(r'browse/([^/]+)', start_url)
        if court_match:
            court_name = court_match.group(1)
        else:
            court_name = "supreme_court"  # Default to Supreme Court
        
        # Get current year and last year
        current_year = datetime.now().year
        years_to_try = [current_year, current_year-1, current_year-2]
        
        for year in years_to_try:
            if len(judgment_links) >= limit:
                break
                
            # Different URL patterns to try
            urls_to_try = [
                f"https://indiankanoon.org/search/?formInput=doctypes:{court_name}%20fromdate:1-1-{year}%20todate:%2031-12-{year}",
                f"https://indiankanoon.org/browse/{court_name}/{year}/",
                f"https://indiankanoon.org/browse/{court_name}/?year={year}"
            ]
            
            for url in urls_to_try:
                print(f"Trying to fetch judgments from: {url}")
                
                # Use a more browser-like approach
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0',
                    'TE': 'Trailers',
                    'Pragma': 'no-cache'
                }
                
                try:
                    # Add a longer delay before request to avoid rate limiting
                    time.sleep(5 + random.uniform(2, 5))
                    response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract document links
                        doc_links = []
                        for a in soup.find_all('a', href=True):
                            if '/doc/' in a['href']:
                                full_url = urljoin(url, a['href'])
                                if full_url not in judgment_links:
                                    doc_links.append(full_url)
                        
                        print(f"Found {len(doc_links)} document links on {url}")
                        judgment_links.extend(doc_links[:limit - len(judgment_links)])
                        
                        if len(judgment_links) >= limit:
                            break
                    else:
                        print(f"Failed to access {url} - Status code: {response.status_code}")
                        
                except Exception as e:
                    print(f"Error accessing {url}: {str(e)}")
                    time.sleep(3)  # Pause before trying next URL
                    continue
                
                # Add a delay between attempts
                time.sleep(3 + random.uniform(1, 3))
        
        # Process collected judgment links
        print(f"Total Indian Kanoon judgment links collected: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing judgment {pages_processed+1}/{limit}: {url}")
            
            # Add substantial delay between document requests
            time.sleep(7 + random.uniform(3, 6))
            
            try:
                # Use a more browser-like approach with different user agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Referer': 'https://indiankanoon.org/browse/',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    content = BeautifulSoup(response.text, 'html.parser')
                    self.pipeline.process_url(url, content)
                    pages_processed += 1
                else:
                    print(f"Failed to access document {url} - Status code: {response.status_code}")
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
            
        print(f"Completed Indian Kanoon crawl. Processed {pages_processed} pages.")
        
    def crawl_sci_judgments(self, year=None, limit=20):
        """Crawl Supreme Court of India judgments with support for multiple years"""
        if year is None:
            year = datetime.now().year
            
        print(f"Starting Supreme Court judgment crawl for year: {year}")
        
        # The Supreme Court changed their website structure
        # Try several possible URL patterns for judgments
        possible_urls = [
            f"https://main.sci.gov.in/judgments/{year}",
            f"https://main.sci.gov.in/judgment/{year}",
            f"https://main.sci.gov.in/supremecourt/judgments/{year}",
            f"https://main.sci.gov.in/judgement/{year}",
            f"https://www.sci.gov.in/judgments",
            f"https://www.sci.gov.in/judgment-order"
        ]
        
        judgment_links = []
        content = None
        
        # Try each possible URL pattern
        for url in possible_urls:
            print(f"Trying to access Supreme Court judgments at: {url}")
            try:
                # Use a more browser-like approach
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    content = BeautifulSoup(response.text, 'html.parser')
                    print(f"Successfully accessed: {url}")
                    break
                else:
                    print(f"Failed to access {url} - Status code: {response.status_code}")
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
            
            # Add delay between attempts
            time.sleep(3 + random.uniform(1, 2))
        
        if not content:
            print("Failed to access any Supreme Court judgment listing page")
            return
            
        # Find all judgment links
        for a in content.find_all('a', href=True):
            link = a['href']
            link_text = a.get_text().lower()
            
            # Look for PDF links or judgment page links
            if (link.endswith('.pdf') or '/judgment/' in link or 
                'judgment' in link_text or 'judgement' in link_text):
                try:
                    full_url = urljoin(url, link)
                    if full_url not in judgment_links:
                        judgment_links.append(full_url)
                except Exception:
                    continue
                
        print(f"Found {len(judgment_links)} judgment links")
        
        # If we didn't find enough, try previous year as well
        if len(judgment_links) < limit and year > 2015:
            print(f"Looking for more judgments in year {year-1}")
            
            # Try the previous year with the URL pattern that worked
            prev_year_url = url.replace(str(year), str(year-1))
            try:
                response = requests.get(prev_year_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    prev_year_content = BeautifulSoup(response.text, 'html.parser')
                    
                    for a in prev_year_content.find_all('a', href=True):
                        link = a['href']
                        if link.endswith('.pdf') or '/judgment/' in link:
                            try:
                                full_url = urljoin(prev_year_url, link)
                                if full_url not in judgment_links:
                                    judgment_links.append(full_url)
                            except Exception:
                                continue
                    
                    print(f"Total judgment links after including previous year: {len(judgment_links)}")
            except Exception as e:
                print(f"Error accessing previous year page: {str(e)}")
                
        # Process up to the limit
        pages_processed = 0
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing judgment {pages_processed+1}/{limit}: {url}")
            
            try:
                # PDFs need to be handled differently than HTML pages
                if url.lower().endswith('.pdf'):
                    # Use requests to get PDF content directly
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code == 200:
                        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                            text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
                            self.pipeline.process_url(url, text)
                            pages_processed += 1
                    else:
                        print(f"Failed to download PDF: {url}")
                else:
                    # For HTML pages, use regular processing
                    content = self.scraper.fetch_content(url)
                    if content:
                        self.pipeline.process_url(url, content)
                        pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
            
            # Respect rate limits with randomization
            time.sleep(3 + random.uniform(2, 5))
            
        print(f"Completed Supreme Court crawl. Processed {pages_processed} judgments.")
        
    def crawl_high_court(self, court_url, limit=20):
        """Crawl a specific High Court's judgments using direct search and document links"""
        print(f"Starting High Court crawl using search approach")
        pages_processed = 0
        judgment_links = []
        
        # Extract court name from the URL
        court_match = re.search(r'browse/([^/]+)', court_url)
        if court_match:
            court_name = court_match.group(1)
        else:
            # Try to extract court name from domain
            domain = urlparse(court_url).netloc
            for hc in ['delhi', 'bombay', 'calcutta', 'madras', 'allahabad',
                      'karnataka', 'kerala', 'punjab', 'haryana', 'patna',
                      'gujarat', 'telangana', 'orissa', 'rajasthan']:
                if hc in domain or hc in court_url.lower():
                    court_name = f"{hc}_hc"
                    break
            else:
                court_name = "delhi_hc"  # Default
        
        # Get current year and previous years
        current_year = datetime.now().year
        years_to_try = [current_year, current_year-1, current_year-2]
        
        # Similar to IndianKanoon crawler, try multiple URL patterns
        for year in years_to_try:
            if len(judgment_links) >= limit:
                break
                
            # Different URL patterns to try
            urls_to_try = []
            
            # Check if this is an IndianKanoon URL
            if 'indiankanoon.org' in court_url:
                urls_to_try = [
                    f"https://indiankanoon.org/search/?formInput=doctypes:{court_name}%20fromdate:1-1-{year}%20todate:%2031-12-{year}",
                    f"https://indiankanoon.org/browse/{court_name}/{year}/",
                    f"https://indiankanoon.org/browse/{court_name}/?year={year}"
                ]
            else:
                # For official High Court websites
                court_domain = urlparse(court_url).netloc
                
                # For DHC
                if 'delhi' in court_name or 'delhi' in court_url.lower():
                    urls_to_try = [
                        f"http://dhcsc.nic.in/search/?formInput=fromdate:1-1-{year}%20todate:31-12-{year}",
                        f"http://dhcsc.nic.in/judgments/{year}/",
                        f"http://delhihighcourt.nic.in/judgments/{year}"
                    ]
                # For other High Courts, try general patterns
                else:
                    urls_to_try = [
                        f"{court_url}/judgments/{year}",
                        f"{court_url}/judgment/{year}",
                        f"{court_url}/judgements/{year}"
                    ]
            
            for url in urls_to_try:
                print(f"Trying to fetch judgments from: {url}")
                
                # Use a more browser-like approach
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                try:
                    # Add a longer delay before request to avoid rate limiting
                    time.sleep(5 + random.uniform(2, 4))
                    response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract document links
                        doc_links = []
                        
                        # Look for both judgment PDFs and document pages
                        for a in soup.find_all('a', href=True):
                            link = a['href']
                            if ('/doc/' in link or 
                                link.lower().endswith('.pdf') or 
                                '/judgment/' in link or
                                '/judgement/' in link or
                                'caseno=' in link):
                                
                                full_url = urljoin(url, link)
                                if full_url not in judgment_links and full_url not in doc_links:
                                    doc_links.append(full_url)
                        
                        print(f"Found {len(doc_links)} document links on {url}")
                        judgment_links.extend(doc_links[:limit - len(judgment_links)])
                        
                        if len(judgment_links) >= limit:
                            break
                    else:
                        print(f"Failed to access {url} - Status code: {response.status_code}")
                        
                except Exception as e:
                    print(f"Error accessing {url}: {str(e)}")
                    time.sleep(3)  # Pause before trying next URL
                    continue
                
                # Add a delay between attempts
                time.sleep(3 + random.uniform(1, 3))
        
        # Process collected judgment links
        print(f"Total High Court judgment links collected: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing High Court judgment {pages_processed+1}/{limit}: {url}")
            
            # Add delay between document requests
            time.sleep(6 + random.uniform(2, 4))
            
            try:
                # Handle PDF links directly
                if url.lower().endswith('.pdf'):
                    # Use requests to get PDF content directly
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code == 200:
                        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                            text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
                            self.pipeline.process_url(url, text)
                            pages_processed += 1
                    else:
                        print(f"Failed to download PDF: {url}")
                else:
                    # For HTML pages, use our scraper
                    content = None
                    
                    # For indiankanoon.org, use direct requests to avoid 403
                    if 'indiankanoon.org' in url:
                        response = requests.get(url, headers=headers, timeout=15)
                        if response.status_code == 200:
                            content = BeautifulSoup(response.text, 'html.parser')
                    else:
                        content = self.scraper.fetch_content(url)
                        
                    if content:
                        self.pipeline.process_url(url, content)
                        pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
            
        print(f"Completed High Court crawl. Processed {pages_processed} pages.")
    
    def crawl_livelaw(self, limit=20):
        """Crawl LiveLaw.in judgments with improved extraction"""
        base_urls = [
            "https://www.livelaw.in/tags/judgments/",
            "https://www.livelaw.in/high-court/allahabad-high-court",
            "https://www.livelaw.in/high-court/bombay-high-court",
            "https://www.livelaw.in/high-court/delhi-high-court",
            "https://www.livelaw.in/high-court/karnataka-high-court",
            "https://www.livelaw.in/high-court/calcutta-high-court",
            "https://www.livelaw.in/supreme-court/"
        ]
        domain = "https://www.livelaw.in"
        print(f"Starting LiveLaw judgment crawl (target: {limit} judgments)")
        pages_processed = 0
        judgment_links = []
        
        # First: Gather links from all category/listing pages
        pages_per_source = max(3, int(limit / len(base_urls))) + 1
        
        for base_url in base_urls:
            if pages_processed >= limit:
                break
                
            print(f"Fetching listing page: {base_url}")
            content = self.scraper.fetch_content(base_url)
            if not content or not isinstance(content, BeautifulSoup):
                print(f"Failed to fetch LiveLaw listing page: {base_url}")
                continue
                
            # Extract the first batch of articles
            self._extract_livelaw_articles(content, judgment_links, domain)
            
            # Look for pagination links to get more articles
            pagination_links = []
            for a in content.find_all('a', href=True):
                if 'page' in a['href'] and a['href'] not in pagination_links:
                    # This is likely a pagination link
                    full_url = urljoin(base_url, a['href'])
                    if full_url not in pagination_links:
                        pagination_links.append(full_url)
            
            # Process some of the pagination links
            for page_url in pagination_links[:pages_per_source]:
                print(f"Fetching pagination page: {page_url}")
                page_content = self.scraper.fetch_content(page_url)
                if page_content and isinstance(page_content, BeautifulSoup):
                    self._extract_livelaw_articles(page_content, judgment_links, domain)
                
                # Respect rate limits
                time.sleep(1 + random.uniform(0.5, 1.5))
            
            print(f"Total judgment links found so far: {len(judgment_links)}")
            
            # Respect rate limits between categories
            time.sleep(2 + random.uniform(1, 3))
                    
        print(f"Found {len(judgment_links)} potential judgment links on LiveLaw")
        
        # Second: Filter and deduplicate judgment links
        filtered_links = []
        seen_titles = set()
        
        for link in judgment_links:
            # Skip category/listing pages
            if any(x in link for x in base_urls) or link.endswith('/tags/judgments/'):
                continue
                
            # Skip digest/summary pages based on URL patterns
            if any(x in link.lower() for x in ['digest', 'round-up', 'weekly-round', 'monthly-digest', 'monthly-round']):
                continue
            
            # Skip if we've seen a similar URL (title part)
            url_parts = urlparse(link).path.split('/')
            if len(url_parts) > 1:
                title_part = url_parts[-1]
                if title_part in seen_titles:
                    continue
                seen_titles.add(title_part)
            
            filtered_links.append(link)
            
        print(f"Filtered down to {len(filtered_links)} likely unique judgment links")
        
        # Shuffle links to get a better variety of courts and topics
        random.shuffle(filtered_links)
        
        # Third: Process individual judgment pages
        for url in filtered_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing LiveLaw judgment {pages_processed+1}/{limit}: {url}")
            
            # Fetch page content
            content = self.scraper.fetch_content(url)
            if not content or not isinstance(content, BeautifulSoup):
                print(f"Failed to fetch {url}")
                continue
            
            # Final check - verify this looks like an actual judgment page before processing
            if self._is_actual_judgment_page(content, url):
                result = self.pipeline.process_url(url, content)
                if result:
                    pages_processed += 1
            else:
                print(f"Skipping non-judgment article: {url}")
            
            # Respect rate limits with randomization
            time.sleep(2 + random.uniform(1, 2))
            
            if pages_processed >= limit:
                break
            
        print(f"Completed LiveLaw crawl. Processed {pages_processed} individual judgments.")
        
    def _extract_livelaw_articles(self, soup, judgment_links, domain):
        """Extract article links from LiveLaw pages"""
        # Find articles in various container formats
        articles = soup.find_all('article')
        if not articles:
            # Try alternative class names for articles/posts
            articles = soup.find_all(['div', 'li'], {'class': ['post', 'entry', 'article-item', 
                                                               'td_module_16', 'td_module_wrap',
                                                               'tdb_module_loop', 'td-module-container']})
        
        if articles:
            print(f"Found {len(articles)} article listings")
            for article in articles:
                # Get the main article link
                article_link = article.find('a', href=True)
                if article_link:
                    link = article_link['href']
                    # Convert relative URLs to absolute URLs
                    if link.startswith('/'):
                        link = domain + link
                    if link not in judgment_links and 'livelaw.in' in link:
                        # Filter links that likely contain judgments
                        if any(x in link for x in ['/top-stories/', '/judgments/', 'read-judgment', 'supreme-court', 'high-court']):
                            judgment_links.append(link)
        else:
            # Fallback to looking for any links that look like judgment articles
            for a in soup.find_all('a', href=True):
                link = a['href']
                link_text = a.get_text().lower().strip()
                
                if link.startswith('/'):
                    link = domain + link
                
                # Check for judgment indicators in link or text
                is_judgment = any(x in link.lower() for x in ['judgment', 'supreme-court', 'high-court']) and \
                              any(x in link_text.lower() for x in ['read', 'full', 'judgment', 'supreme', 'court', 'justice'])
                
                if is_judgment and link not in judgment_links and 'livelaw.in' in link:
                    judgment_links.append(link)
                    
        return len(judgment_links)
        
    def _is_actual_judgment_page(self, soup, url):
        """Determine if a page actually contains a judgment rather than a digest/summary"""
        # Check page title
        if soup.title:
            title = soup.title.get_text().lower()
            if any(x in title for x in ['digest', 'round up', 'weekly round', 'monthly digest']):
                return False
                
        # Check for judgment indicators in the content
        main_text = soup.get_text().lower()
        judgment_indicators = [
            'read judgment', 'read the judgment', 'full judgment', 
            'judgment authored by', 'authored the judgment',
            'court held', 'court observed', 'court ruled',
            'bench comprising', 'coram:', 'versus'
        ]
        
        judgment_indicator_count = sum(1 for indicator in judgment_indicators if indicator in main_text)
        
        # Look for embedded PDF or judgment text
        judgment_embedding = (
            soup.find('iframe', {'class': 'judgement-embed'}) or
            soup.find('div', {'class': 'judgement-embed'}) or
            soup.find('div', {'class': ['judgement-content', 'judgment-content']})
        )
        
        # Return true if we have strong indicators this is an actual judgment
        return judgment_indicator_count >= 2 or judgment_embedding is not None
    
    def crawl_official_high_court(self, court_url, court_name, limit=20):
        """Crawl official High Court websites that have different structures from Indian Kanoon"""
        print(f"Starting official {court_name} High Court crawl")
        pages_processed = 0
        judgment_links = []
        
        # Current year and previous years
        current_year = datetime.now().year
        years_to_try = [current_year, current_year-1, current_year-2]
        
        for year in years_to_try:
            # Try different URL patterns based on the court
            if "delhi" in court_name.lower():
                urls_to_try = [
                    f"{court_url}/dhc/home/judgments/{year}",
                    f"{court_url}/judgments/{year}",
                    f"{court_url}/dhcqrydisp_o.asp?pYear={year}"
                ]
            elif "bombay" in court_name.lower():
                urls_to_try = [
                    f"{court_url}/judgements/{year}",
                    f"{court_url}/judgments/{year}",
                    f"{court_url}/judgementlist.aspx?yr={year}"
                ]
            elif "calcutta" in court_name.lower():
                urls_to_try = [
                    f"{court_url}/judgments/{year}",
                    f"{court_url}/judgement/judg_{year}.php",
                    f"{court_url}/judgements/{year}"
                ]
            else:
                urls_to_try = [
                    f"{court_url}/judgments/{year}",
                    f"{court_url}/judgement/{year}",
                    f"{court_url}/judgements/{year}",
                    f"{court_url}/cases/{year}"
                ]
            
            # Try each URL pattern
            for url in urls_to_try:
                print(f"Trying to access: {url}")
                
                # Use a rotating set of headers
                headers = {
                    'User-Agent': random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
                    ]),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.google.com/',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                try:
                    # Add delay to avoid rate limiting
                    time.sleep(4 + random.uniform(1, 3))
                    response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract links to judgment PDFs or pages
                        doc_links = []
                        for a in soup.find_all('a', href=True):
                            link = a['href']
                            link_text = a.get_text().lower()
                            
                            # Check if this looks like a judgment link
                            if (link.endswith('.pdf') or 
                                '/judgment' in link.lower() or 
                                'judgement' in link.lower() or
                                'ord_' in link.lower() or
                                'jud_' in link.lower() or
                                ('case' in link_text and ('no' in link_text or 'judgment' in link_text))):
                                
                                full_url = urljoin(url, link)
                                if full_url not in judgment_links:
                                    doc_links.append(full_url)
                        
                        print(f"Found {len(doc_links)} potential judgment links on {url}")
                        judgment_links.extend(doc_links[:limit - len(judgment_links)])
                        
                        if len(judgment_links) >= limit:
                            break
                    else:
                        print(f"Failed to access {url} - Status code: {response.status_code}")
                
                except Exception as e:
                    print(f"Error accessing {url}: {str(e)}")
                    continue
            
            if len(judgment_links) >= limit:
                break
        
        # Process collected judgment links
        print(f"Total {court_name} High Court judgment links: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing {court_name} judgment {pages_processed+1}/{limit}: {url}")
            
            # Add delay between requests
            time.sleep(3 + random.uniform(2, 4))
            
            try:
                if url.lower().endswith('.pdf'):
                    # Handle PDF directly
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code == 200:
                        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                            text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
                            self.pipeline.process_url(url, text)
                            pages_processed += 1
                    else:
                        print(f"Failed to download PDF: {url}")
                else:
                    # Process HTML content
                    content = self.scraper.fetch_content(url)
                    if content:
                        self.pipeline.process_url(url, content)
                        pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        print(f"Completed {court_name} High Court crawl. Processed {pages_processed} judgments.")
    
    def crawl_legal_service_india(self, limit=20):
        """Crawl Legal Services India which has a collection of judgments"""
        print("Starting Legal Services India crawl")
        pages_processed = 0
        judgment_links = []
        
        # URLs to try for Legal Services India
        urls_to_try = [
            "https://www.legalserviceindia.com/supreme-court/judgments.htm",
            "https://www.legalserviceindia.com/legal/supreme-court-judgments.html",
            "https://www.legalserviceindia.com/legal/high-court-judgments.html",
            "https://www.legalserviceindia.com/legal/category-27-high-court-judgments.html",
            "https://www.legalserviceindia.com/legal/category-8-supreme-court-judgments.html"
        ]
        
        for url in urls_to_try:
            print(f"Trying to access: {url}")
            
            try:
                # Add delay
                time.sleep(3 + random.uniform(1, 2))
                
                # Use a different user agent for each request
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for links to individual judgment pages
                    for a in soup.find_all('a', href=True):
                        link = a['href']
                        link_text = a.get_text().lower()
                        
                        # Check if this could be a judgment page
                        if (any(term in link_text for term in ['vs', 'versus', 'v.', 'judgment']) and
                            (link.startswith('http') or link.startswith('/legal/'))):
                            
                            full_url = urljoin(url, link)
                            if full_url not in judgment_links:
                                judgment_links.append(full_url)
                    
                    print(f"Found {len(judgment_links)} judgment links on {url}")
                    
                    if len(judgment_links) >= limit:
                        break
                else:
                    print(f"Failed to access {url} - Status code: {response.status_code}")
            
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
                continue
        
        # Process judgment links
        print(f"Total Legal Services India judgment links: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing judgment {pages_processed+1}/{limit}: {url}")
            
            # Add delay
            time.sleep(3 + random.uniform(1, 3))
            
            try:
                content = self.scraper.fetch_content(url)
                if content:
                    self.pipeline.process_url(url, content)
                    pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        print(f"Completed Legal Services India crawl. Processed {pages_processed} judgments.")
    
    def crawl_india_law(self, limit=20):
        """Crawl India Law website which has case laws and judgments"""
        print("Starting India Law crawl")
        pages_processed = 0
        judgment_links = []
        
        # URLs to try for India Law
        urls_to_try = [
            "https://www.indialaw.in/blog/case-laws/",
            "https://www.indialaw.in/blog/case-laws/page/1/",
            "https://www.indialaw.in/blog/case-laws/page/2/",
            "https://www.indialaw.in/blog/case-laws/supreme-court-cases/",
            "https://www.indialaw.in/blog/case-laws/high-court-cases/"
        ]
        
        for url in urls_to_try:
            print(f"Trying to access: {url}")
            
            try:
                # Add delay
                time.sleep(3 + random.uniform(1, 2))
                
                # Use a different user agent for each request
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for article elements that contain case law
                    articles = soup.find_all(['article', 'div'], {'class': ['post', 'entry', 'article']})
                    
                    # If no article containers found, try to find links directly
                    if not articles:
                        for a in soup.find_all('a', href=True):
                            link = a['href']
                            link_text = a.get_text().lower()
                            
                            # Check if this looks like a case law link
                            if ('case' in link_text or 'judgment' in link_text or 'vs' in link_text) and 'blog/case-laws' in link:
                                full_url = urljoin(url, link)
                                if full_url not in judgment_links:
                                    judgment_links.append(full_url)
                    else:
                        # Extract links from article containers
                        for article in articles:
                            link_elem = article.find('a', href=True)
                            if link_elem:
                                full_url = urljoin(url, link_elem['href'])
                                if full_url not in judgment_links:
                                    judgment_links.append(full_url)
                    
                    print(f"Found {len(judgment_links)} judgment links on {url}")
                    
                    if len(judgment_links) >= limit:
                        break
                else:
                    print(f"Failed to access {url} - Status code: {response.status_code}")
            
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
                continue
        
        # Process judgment links
        print(f"Total India Law judgment links: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing judgment {pages_processed+1}/{limit}: {url}")
            
            # Add delay
            time.sleep(3 + random.uniform(1, 3))
            
            try:
                content = self.scraper.fetch_content(url)
                if content:
                    self.pipeline.process_url(url, content)
                    pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        print(f"Completed India Law crawl. Processed {pages_processed} judgments.")
        
    def crawl_latestlaws(self, limit=20):
        """Crawl LatestLaws.com which has Indian judgments"""
        print("Starting LatestLaws crawl")
        pages_processed = 0
        judgment_links = []
        
        # URLs to try for LatestLaws
        urls_to_try = [
            "https://www.latestlaws.com/latest-caselaw/supreme-court-cases/",
            "https://www.latestlaws.com/latest-caselaw/high-court-cases/",
            "https://www.latestlaws.com/latest-caselaw/supreme-court-cases/2023-latest-caselaw/",
            "https://www.latestlaws.com/latest-caselaw/supreme-court-cases/2022-latest-caselaw/",
            "https://www.latestlaws.com/latest-caselaw/high-court-cases/delhi-high-court-cases/",
            "https://www.latestlaws.com/latest-caselaw/high-court-cases/bombay-high-court-cases/"
        ]
        
        for url in urls_to_try:
            print(f"Trying to access: {url}")
            
            try:
                # Add delay
                time.sleep(3 + random.uniform(1, 2))
                
                # Use a different user agent for each request
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/'
                }
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for case links
                    for a in soup.find_all('a', href=True):
                        link = a['href']
                        link_text = a.get_text().lower()
                        
                        # Check if this is a case link
                        if ('vs' in link_text or 'versus' in link_text) and ('latest-caselaw' in link or 'sc-cases' in link or 'hc-cases' in link):
                            full_url = urljoin(url, link)
                            if full_url not in judgment_links:
                                judgment_links.append(full_url)
                    
                    print(f"Found {len(judgment_links)} judgment links on {url}")
                    
                    if len(judgment_links) >= limit:
                        break
                else:
                    print(f"Failed to access {url} - Status code: {response.status_code}")
            
            except Exception as e:
                print(f"Error accessing {url}: {str(e)}")
                continue
        
        # Process judgment links
        print(f"Total LatestLaws judgment links: {len(judgment_links)}")
        for url in judgment_links[:limit]:
            if url in self.visited:
                continue
                
            self.visited.add(url)
            print(f"Processing judgment {pages_processed+1}/{limit}: {url}")
            
            # Add delay
            time.sleep(3 + random.uniform(1, 3))
            
            try:
                content = self.scraper.fetch_content(url)
                if content:
                    self.pipeline.process_url(url, content)
                    pages_processed += 1
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        print(f"Completed LatestLaws crawl. Processed {pages_processed} judgments.")

# Example Usage
if __name__ == "__main__":
    # Create data dir if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    pipeline = LegalDataPipeline()
    crawler = IndianLegalCrawler(pipeline)
    
    print("Starting Legal Web Scraper")
    print("=" * 50)
    print("This will crawl multiple legal websites to create a comprehensive dataset")
    print("Data will be organized by source for ML training purposes")
    
    # Crawl multiple sources to build a comprehensive dataset
    sources_to_crawl = [
        {
            "name": "LiveLaw Supreme Court",
            "function": crawler.crawl_livelaw,
            "args": [30],  # Process 30 judgments
            "description": "Supreme Court judgments from LiveLaw.in"
        },
        {
            "name": "LiveLaw High Courts", 
            "function": crawler.crawl_livelaw,
            "args": [20],
            "description": "Various High Court judgments from LiveLaw.in"
        },
        {
            "name": "Legal Services India",
            "function": crawler.crawl_legal_service_india,
            "args": [20],
            "description": "Judgments from Legal Services India"
        },
        {
            "name": "India Law",
            "function": crawler.crawl_india_law,
            "args": [20],
            "description": "Case laws from IndiaLaw.in"
        },
        {
            "name": "LatestLaws",
            "function": crawler.crawl_latestlaws,
            "args": [20],
            "description": "Supreme Court and High Court judgments from LatestLaws.com"
        },
        {
            "name": "Delhi High Court Official",
            "function": crawler.crawl_official_high_court,
            "args": ["http://delhihighcourt.nic.in", "Delhi", 15],
            "description": "Judgments from Delhi High Court official website"
        },
        {
            "name": "Bombay High Court Official",
            "function": crawler.crawl_official_high_court,
            "args": ["https://bombayhighcourt.nic.in", "Bombay", 15],
            "description": "Judgments from Bombay High Court official website"
        },
        {
            "name": "Calcutta High Court Official",
            "function": crawler.crawl_official_high_court,
            "args": ["https://calcuttahighcourt.gov.in", "Calcutta", 15],
            "description": "Judgments from Calcutta High Court official website"
        },
        {
            "name": "Supreme Court Official",
            "function": crawler.crawl_sci_judgments,
            "args": [2023, 25],  # Year 2023, 25 judgments
            "description": "2023 Supreme Court judgments from official site"
        }
    ]
    
    # Process each source
    for source in sources_to_crawl:
        try:
            print("\n" + "=" * 50)
            print(f"Crawling: {source['name']}")
            print(f"Description: {source['description']}")
            print("=" * 50)
            
            # Call the appropriate crawler function with its arguments
            source["function"](*source["args"])
            
            # Add a pause between sources to avoid overloading
            pause_time = 10 + random.uniform(5, 15)
            print(f"Pausing for {pause_time:.1f} seconds before next source...")
            time.sleep(pause_time)
            
        except Exception as e:
            print(f"Error crawling {source['name']}: {str(e)}")
            continue
    
    # Generate summary statistics of collected data
    print("\n" + "=" * 50)
    print("Data Collection Summary")
    print("=" * 50)
    
    # Query the database for statistics
    cursor = pipeline.db.cursor()
    
    # Total document count
    cursor.execute("SELECT COUNT(*) FROM judgments")
    total_docs = cursor.fetchone()[0]
    print(f"Total documents collected: {total_docs}")
    
    # Documents by court
    print("\nDocuments by court:")
    cursor.execute("""
        SELECT court, COUNT(*) as count 
        FROM judgments 
        GROUP BY court 
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"- {row[0]}: {row[1]} documents")
    
    # Documents with proper metadata
    print("\nDocuments with complete metadata:")
    cursor.execute("""
        SELECT COUNT(*) FROM judgments 
        WHERE json_extract(metadata, '$.primary_case_title') IS NOT NULL 
        AND json_extract(metadata, '$.primary_citation') IS NOT NULL
    """)
    complete_metadata = cursor.fetchone()[0]
    print(f"- {complete_metadata} documents have complete metadata ({(complete_metadata/total_docs)*100:.1f}%)")
    
    # Recent documents
    print("\nMost recently processed judgments:")
    cursor.execute("""
        SELECT id, title, court, judgment_date 
        FROM judgments 
        ORDER BY created_at DESC 
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"ID: {row[0]}")
        print(f"Title: {row[1]}")
        print(f"Court: {row[2]}")
        print(f"Date: {row[3]}")
        print("-" * 30)
        
    # Check directory structure
    print("\nData organization structure:")
    data_dirs = []
    for root, dirs, files in os.walk("data"):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.json')])
            if file_count > 0:
                print(f"- {dir_path}: {file_count} documents")