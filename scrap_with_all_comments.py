import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
import concurrent.futures

# --- Configuration ---
TAGS_TO_SCRAPE = {
    "Modeling": "Modeling",
    "User Experience": "user-experience",
    "Data Integration": "data-integration",
    "Calculation Functions": "calculation-functions",
    "Importing and Exporting Data": "Importing-and-Exporting-Data",
    # "Security and Administration": "Security-and-Administration",
    # "UX Designer": "UX-Designer",
    # "UX": "UX",
    # "Model Builder": "Model-Builder",
    # "Anaplan Community": "anaplan-community",
    # "Application Lifecycle Management": "application-lifecycle-management",
    # "How to": "How-to",
    # "Extensions": "Extensions",
    # "Anaplan Home": "Anaplan-Home",
    # "Certified Master Anaplanner": "certified-master-anaplanner"
}


MAX_WORKERS = 10
MAX_RETRIES = 3
MAX_PAGES_PER_TAG = {
    # "Modeling": 1,
    "Modeling": 245,
    "User Experience": 111, "Data Integration": 109,
    "Calculation Functions": 89, "Importing and Exporting Data": 77,
    "Security and Administration": 38, "UX Designer": 28, "UX": 25,
    "Model Builder": 15, "Anaplan Community": 12,
    "Application Lifecycle Management": 12,
    # "How to": 1,
    "How to": 11,
    "Extensions": 11,
    "Anaplan Home": 8,
    "Certified Master Anaplanner": 8
}
RETRY_DELAY = 5

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# --- Main Functions ---
def get_threads_from_list_page(page_url):
    """Fetches metadata for all threads on a single list page."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            module = soup.find('div', {'data-react': 'DiscussionListModule'})
            if module and 'data-props' in module.attrs:
                data = json.loads(module['data-props'])
                return data.get('discussions', [])
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [!] Failed to fetch list page {page_url}: {e}")
    return []


def scrape_thread_page(thread_meta):
    """
    Scrapes a single discussion thread page, handling different comment structures.
    This function is designed to be run by a worker thread.
    """
    thread_url = thread_meta.get('url')
    if not thread_url: return None

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.5, 2.0))
            response = requests.get(thread_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            main_column = soup.find('div', class_='mainColumn')
            if not main_column: return None

            thread_data = {
                "question_title": (main_column.find('h1').get_text(strip=True) if main_column.find('h1') else ''),
                "question_url": thread_url, "question_author": None, "question_body": None,
                "tags": [], "accepted_answers": [], "other_comments": []
            }

            author_tag = main_column.find('a', class_='seoUser')
            if author_tag:
                author_name_tag = author_tag.find('span', class_='seoUserName')
                if author_name_tag:
                    thread_data["question_author"] = author_name_tag.get_text(strip=True)

            question_body_element = main_column.find('div', class_='userContent')
            if question_body_element:
                thread_data["question_body"] = question_body_element.get_text(strip=True)

            tags_heading = main_column.find('h2', string='Tags')
            if tags_heading:
                tags_container = tags_heading.find_parent('div', class_='pageBox')
                if tags_container:
                    all_tag_links = tags_container.find_all('a', href=lambda href: href and 'tagID' in href)
                    thread_data["tags"] = [tag.get_text(strip=True) for tag in all_tag_links]

            accepted_texts = set()
            accepted_heading = main_column.find('h2', string='Accepted answers')
            if accepted_heading:
                for sibling in accepted_heading.find_next_siblings():
                    if sibling.name == 'h2': break
                    if 'comment' in sibling.get('class', []):
                        author_tag = sibling.find('span', class_='seoUserName')
                        author = author_tag.get_text(strip=True) if author_tag else "N/A"
                        answer_body_tag = sibling.find('div', class_='userContent')
                        answer_body = answer_body_tag.get_text(strip=True) if answer_body_tag else ""
                        if answer_body:
                            thread_data["accepted_answers"].append({"author": author, "text": answer_body})
                            accepted_texts.add(answer_body)

            comments_heading = main_column.find('h2', string=['All comments', 'Comments'])
            if comments_heading:
                for sibling in comments_heading.find_next_siblings():
                    if sibling.name == 'h2': break
                    if 'comment' in sibling.get('class', []):
                        author_tag = sibling.find('span', class_='seoUserName')
                        author = author_tag.get_text(strip=True) if author_tag else "N/A"
                        comment_body_tag = sibling.find('div', class_='userContent')
                        comment_body = comment_body_tag.get_text(strip=True) if comment_body_tag else ""

                        if comment_body and comment_body not in accepted_texts:
                            thread_data["other_comments"].append({"author": author, "text": comment_body})

            thread_data['views'] = thread_meta.get('countViews', 0)
            thread_data['comments_count'] = thread_meta.get('countComments', 0)
            return thread_data

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [!] Error scraping {thread_url}. Retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))

    print(f"    [!] Failed to scrape thread {thread_url} after {MAX_RETRIES} attempts. Skipping.")
    return None


# --- Script Execution ---
if __name__ == "__main__":
    print("🚀 Starting Parallel Anaplan Community Scraper...")

    for tag_name, tag_url_slug in TAGS_TO_SCRAPE.items():
        max_pages = MAX_PAGES_PER_TAG.get(tag_name, 1)
        output_filename = f'anaplan_{tag_url_slug}_discussions.json'
        # output_filename = f'anaplan_comments_check_discussions.json'
        print(f"\n--- Processing Tag: '{tag_name}' (up to {max_pages} pages) ---")
        start_time = time.time()
        print(f"Time Started : {start_time:.2f}")

        all_threads_to_scrape = []
        for page_num in range(1, max_pages + 1):
            page_url = f"https://community.anaplan.com/discussions/tagged/{tag_url_slug}/p{page_num}"
            print(f"  Gathering URLs from page {page_num}/{max_pages}...", end='\r')
            threads_on_page = get_threads_from_list_page(page_url)
            if not threads_on_page: break
            all_threads_to_scrape.extend(threads_on_page)

        print(f"\n✅ Found a total of {len(all_threads_to_scrape)} threads for this tag.")

        tag_specific_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_thread = {executor.submit(scrape_thread_page, meta): meta for meta in all_threads_to_scrape}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_thread)):
                try:
                    result = future.result()
                    if result: tag_specific_data.append(result)
                    print(f"  Processed {i + 1}/{len(all_threads_to_scrape)} threads...", end='\r')
                except Exception as exc:
                    print(f'  [!] A thread generated an exception: {exc}')
        end_time = time.time()
        print(f"Time Ended : {end_time:.2f}")
        total_time = end_time - start_time
        print(f"Total Time Taken for {tag_name} : {total_time:.6f}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(tag_specific_data, f, indent=4)

        print(f"\n✅ Tag '{tag_name}' complete. Data saved to '{output_filename}'")
        print(f"Total discussions scraped for this tag: {len(tag_specific_data)}")

    print("\n\n🎉 All specified tags have been scraped successfully!")
