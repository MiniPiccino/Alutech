#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_alutech.py

Crawls one or MANY start URLs, obeys robots.txt, extracts page title, meta, visible text,
emails and phone numbers, and writes output .txt files suitable for chatbot ingestion.

USAGE (single URL):
    python scrape_alutech.py --start-url https://alutech.hr/ --output alutech.txt

USAGE (list of URLs from file; one URL per line):
    python scrape_alutech.py --start-list start_urls.txt --output "scrape_{domain}.txt"

Notes:
- When using --start-list, the --output value can include placeholders:
  {i} (1-based index), {host} (full hostname), {domain} (registered domain like 'alutech.hr').
  Example: --output "out/{i:02d}_{domain}.txt"
- Crawling is sequential (polite). Set --delay to control inter-request delay.
"""

import argparse
import time
import re
import os
import urllib.parse
import requests
from bs4 import BeautifulSoup
from urllib import robotparser
from collections import deque
import tldextract

USER_AGENT = "Mozilla/5.0 (compatible; MyScraperBot/1.1; +https://example.com/bot)"
REQUESTS_TIMEOUT = 15
DEFAULT_DELAY_BETWEEN_REQUESTS = 1.0  # seconds, be polite

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
    text = soup.get_text(separator='\n', strip=True)
    # Collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text

def extract_links(soup, base_url):
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href:
            continue
        if href.startswith(('mailto:', 'tel:', 'javascript:')):
            continue
        try:
            full = normalize_url(base_url, href)
            links.add(full)
        except Exception:
            pass
    return links

def safe_content_type(resp):
    return (resp.headers.get('Content-Type') or '').split(';')[0].strip().lower()

def build_output_name(out_template: str, start_url: str, idx: int):
    """
    Supports placeholders:
      {i}      -> 1-based index
      {host}   -> full hostname (e.g. 'www.alutech.hr')
      {domain} -> registered domain + suffix (e.g. 'alutech.hr')
    You can also use format specs like {i:02d}
    """
    parsed = urllib.parse.urlparse(start_url)
    host = parsed.netloc
    ext = tldextract.extract(start_url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    try:
        name = out_template.format(i=idx, host=host, domain=domain)
    except Exception:
        # if user passed plain filename without placeholders
        name = out_template
    # Ensure directory exists if a path is given
    parent = os.path.dirname(name)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return name

def scrape(start_url, output_file, max_pages=200, max_depth=3, delay_between_requests=DEFAULT_DELAY_BETWEEN_REQUESTS):
    rp = get_robots_parser(start_url)
    visited = set()
    q = deque()
    q.append((start_url, 0))
    results = []

    headers = {'User-Agent': USER_AGENT}

    session = requests.Session()
    session.headers.update(headers)

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
            resp = session.get(url, timeout=REQUESTS_TIMEOUT, allow_redirects=True)
            # politeness
            time.sleep(delay_between_requests)

            ctype = safe_content_type(resp)
            if resp.status_code != 200 or ctype != 'text/html':
                visited.add(url)
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            title = (soup.title.string.strip() if soup.title and soup.title.string else '')[:400]
            meta_desc = ''
            desc_tag = soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'property':'og:description'})
            if desc_tag and desc_tag.get('content'):
                meta_desc = desc_tag['content'].strip()

            body_text = extract_visible_text(soup)

            # Emails
            emails = sorted(set(EMAIL_RE.findall(resp.text)))

            # Phones (clean up a bit)
            raw_phone_matches = PHONE_RE.findall(resp.text)
            cleaned_phones = set()
            for m in raw_phone_matches:
                # m is a tuple of groups; join and collapse spaces/dots/dashes
                candidate = ''.join(m)
                candidate = re.sub(r'[\s\.\-]+', ' ', candidate).strip()
                digits = re.sub(r'\D', '', candidate)
                if len(digits) >= 6:
                    cleaned_phones.add(candidate)
            phones = sorted(cleaned_phones)

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

    # Write to one .txt file with clear separators (easy to split for chatbot ingestion)
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
            f.write(page['text'])
            f.write("\n\n")

    print(f"Done. Wrote {len(results)} pages to {output_file}")

def read_start_list(path: str):
    urls = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            # normalize to absolute URL (require scheme)
            if not re.match(r'^https?://', s, re.I):
                s = 'https://' + s
            urls.append(s.rstrip('/'))
    return urls

def main():
    parser = argparse.ArgumentParser(description='Simple site scraper for chatbot ingestion.')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--start-url', help='Starting URL (e.g. https://alutech.hr/)')
    g.add_argument('--start-list', help='Path to a file with multiple start URLs (one per line).')
    parser.add_argument('--output', default='output_{domain}.txt',
                        help='Output filename (supports {i}, {host}, {domain}). Default: output_{domain}.txt')
    parser.add_argument('--max-pages', type=int, default=200, help='Max pages to crawl per start URL')
    parser.add_argument('--max-depth', type=int, default=3, help='Max link depth from each start URL')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY_BETWEEN_REQUESTS,
                        help='Delay between HTTP requests in seconds (politeness)')
    args = parser.parse_args()

    if args.start_url:
        out = build_output_name(args.output, args.start_url, idx=1)
        scrape(args.start_url.rstrip('/'), out, max_pages=args.max_pages, max_depth=args.max_depth,
               delay_between_requests=args.delay)
    else:
        urls = read_start_list(args.start_list)
        if not urls:
            print(f"No URLs found in {args.start_list}")
            return
        for i, url in enumerate(urls, start=1):
            out = build_output_name(args.output, url, idx=i)
            print(f"\n=== [{i}/{len(urls)}] Crawling {url} -> {out} ===")
            try:
                scrape(url, out, max_pages=args.max_pages, max_depth=args.max_depth,
                       delay_between_requests=args.delay)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as e:
                print(f"[fatal] {url} -> {e}")

if __name__ == '__main__':
    main()
