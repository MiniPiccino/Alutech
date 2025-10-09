#!/usr/bin/env python3
"""
scrape_alutech.py

Crawls a single domain, obeys robots.txt, extracts page title, meta, visible text,
emails and phone numbers, and writes a single output .txt suitable for chatbot ingestion.

Usage:
    python scrape_alutech.py --start-url https://alutech.hr/ --output alutech_text_for_chatbot.txt
"""

import argparse
import time
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup
from urllib import robotparser
from collections import deque
import tldextract

USER_AGENT = "Mozilla/5.0 (compatible; MyScraperBot/1.0; +https://example.com/bot)"
REQUESTS_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 1.0  # seconds, be polite

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s\-\.]?)?(\(?\d{2,3}\)?[\s\-\.]?)?[\d\s\-\.]{5,15}')

def is_same_domain(start_url, candidate_url):
    s = tldextract.extract(start_url)
    c = tldextract.extract(candidate_url)
    return (s.domain == c.domain and s.suffix == c.suffix)

def get_robots_parser(start_url):
    parsed = urllib.parse.urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt unavailable, default to allowing
        rp = None
    return rp

def can_fetch(rp, url):
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def normalize_url(base, link):
    return urllib.parse.urljoin(base, link.split('#')[0]).rstrip('/')

def extract_visible_text(soup):
    # Remove script/style/navigation/footer/hidden elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'iframe']):
        tag.decompose()
    # Optionally remove extremely small elements, aria-hidden, or elements with display:none? skip for simplicity
    text = soup.get_text(separator='\n', strip=True)
    # Collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text

def extract_links(soup, base_url):
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href.startswith('mailto:') or href.startswith('tel:'):
            continue
        if href.startswith('javascript:'):
            continue
        try:
            full = normalize_url(base_url, href)
            links.add(full)
        except Exception:
            pass
    return links

def scrape(start_url, output_file, max_pages=200, max_depth=3):
    rp = get_robots_parser(start_url)
    visited = set()
    q = deque()
    q.append((start_url, 0))
    results = []

    headers = {'User-Agent': USER_AGENT}

    while q and len(visited) < max_pages:
        url, depth = q.popleft()
        if url in visited:
            continue
        if not is_same_domain(start_url, url):
            continue
        if not can_fetch(rp, url):
            print(f"[robots] skipping {url}")
            visited.add(url)
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            time.sleep(DELAY_BETWEEN_REQUESTS)
            if resp.status_code != 200 or 'text/html' not in resp.headers.get('Content-Type', ''):
                visited.add(url)
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')

            title = (soup.title.string.strip() if soup.title and soup.title.string else '')[:400]
            meta_desc = ''
            desc_tag = soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'property':'og:description'})
            if desc_tag and desc_tag.get('content'):
                meta_desc = desc_tag['content'].strip()

            body_text = extract_visible_text(soup)
            emails = sorted(set(EMAIL_RE.findall(resp.text)))
            # EMAIL_RE.findall returns list of full matches; ensure dedup and full strings
            # For PHONE_RE, filter nonsense
            phones = sorted(set(m[0] + m[1] for m in PHONE_RE.findall(resp.text) if len(''.join(m)) >= 6))

            results.append({
                'url': url,
                'title': title,
                'meta_description': meta_desc,
                'emails': emails,
                'phones': phones,
                'text': body_text
            })

            visited.add(url)
            print(f"[scraped] ({len(visited)}) {url} (depth {depth})")

            if depth < max_depth:
                for link in extract_links(soup, url):
                    if link not in visited and is_same_domain(start_url, link):
                        q.append((link, depth + 1))

        except requests.RequestException as e:
            print(f"[error] {url} -> {e}")
            visited.add(url)
            continue

    # Write to one .txt file with clear separators (this is easy to split for chatbot ingestion)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, page in enumerate(results, start=1):
            f.write(f"--- PAGE {i} ---\n")
            f.write(f"URL: {page['url']}\n")
            f.write(f"TITLE: {page['title']}\n")
            f.write(f"META_DESCRIPTION: {page['meta_description']}\n")
            if page['emails']:
                f.write("EMAILS: " + ", ".join(page['emails']) + "\n")
            if page['phones']:
                f.write("PHONES: " + ", ".join(page['phones']) + "\n")
            f.write("\n")
            # Optionally truncate very long page text for size control — here we write all of it
            f.write(page['text'])
            f.write("\n\n")

    print(f"Done. Wrote {len(results)} pages to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple site scraper for chatbot ingestion.')
    parser.add_argument('--start-url', required=True, help='Starting URL (e.g. https://alutech.hr/)')
    parser.add_argument('--output', default='output.txt', help='Output .txt filename')
    parser.add_argument('--max-pages', type=int, default=200, help='Max pages to crawl')
    parser.add_argument('--max-depth', type=int, default=3, help='Max link depth from start URL')
    args = parser.parse_args()
    scrape(args.start_url, args.output, max_pages=args.max_pages, max_depth=args.max_depth)
