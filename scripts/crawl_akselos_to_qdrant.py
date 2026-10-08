"""Crawl the Akselos support knowledge base and index it into Qdrant.

Usage:
    uv run python scripts/crawl_akselos_to_qdrant.py [--limit N] [--recreate]
"""

import argparse
import os
import re
import time
import urllib.robotparser
import uuid
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, Tag
from qdrant_client import QdrantClient, models

BASE_URL = "https://support.akselos.com"
SITEMAP_URL = f"{BASE_URL}/support/sitemap.xml"
USER_AGENT = "ai-agent-enterprise-indexer/0.1 (personal knowledge base indexing)"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 1200
REQUEST_DELAY_SECONDS = 1.0
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre")
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Nested blocks are covered by the text of these ancestors.
CONTAINER_TAGS = {"li", "tr", "pre"}


@dataclass
class Article:
    url: str
    title: str
    modified: str
    breadcrumb: list[str]
    sections: list[tuple[str, str]]  # (heading, text)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def discover_article_urls(client: httpx.Client) -> list[str]:
    response = client.get(SITEMAP_URL)
    response.raise_for_status()
    urls = re.findall(r"<loc>([^<]+)</loc>", response.text)
    return sorted({u for u in urls if "/solutions/articles/" in u})


def parse_article(url: str, html: str) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("#article-content")
    title_tag = soup.select_one("h1.fw-page-title")
    if body is None or title_tag is None:
        return None

    meta_p = title_tag.find_next("p")
    modified = clean(meta_p.get_text()) if meta_p else ""
    breadcrumb = [clean(a.get_text()) for a in soup.select("ol.breadcrumbs a")]

    sections: list[tuple[str, list[str]]] = [("", [])]
    for el in body.find_all(BLOCK_TAGS):
        if not isinstance(el, Tag):
            continue
        if any(p.name in CONTAINER_TAGS for p in el.parents if isinstance(p, Tag)):
            continue
        if el.name == "tr":
            text = " | ".join(clean(c.get_text()) for c in el.find_all(["th", "td"]))
        else:
            text = clean(el.get_text(" "))
        if not text:
            continue
        if el.name in HEADING_TAGS:
            sections.append((text, []))
        else:
            sections[-1][1].append(text)

    return Article(
        url=url,
        title=clean(title_tag.get_text()),
        modified=modified,
        breadcrumb=breadcrumb,
        sections=[(h, "\n".join(lines)) for h, lines in sections if lines],
    )


def chunk_article(article: Article) -> list[tuple[str, str]]:
    """Split each section into chunks of at most MAX_CHUNK_CHARS (heading, text)."""
    chunks: list[tuple[str, str]] = []
    for heading, text in article.sections:
        current = ""
        for line in text.split("\n"):
            if current and len(current) + len(line) + 1 > MAX_CHUNK_CHARS:
                chunks.append((heading, current))
                current = ""
            current = f"{current}\n{line}" if current else line
        if current:
            chunks.append((heading, current))
    return chunks


def build_points(article: Article) -> list[models.PointStruct]:
    points = []
    for index, (heading, text) in enumerate(chunk_article(article)):
        context = " > ".join([*article.breadcrumb[2:], article.title, heading]).strip(" >")
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article.url}#{index}")),
                vector=models.Document(text=f"{context}\n{text}", model=EMBEDDING_MODEL),
                payload={
                    "url": article.url,
                    "title": article.title,
                    "section": heading,
                    "breadcrumb": article.breadcrumb,
                    "modified": article.modified,
                    "chunk_index": index,
                    "text": text,
                },
            )
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only crawl the first N articles")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "akselos_support"))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:9000"))
    args = parser.parse_args()

    qdrant = QdrantClient(url=args.qdrant_url)
    if args.recreate and qdrant.collection_exists(args.collection):
        qdrant.delete_collection(args.collection)
    if not qdrant.collection_exists(args.collection):
        qdrant.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )

    total_chunks = 0
    skipped: list[str] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as http:
        # urllib's own fetch is rejected with 403 by this site, which reads as disallow-all.
        robots_response = http.get(f"{BASE_URL}/robots.txt")
        robots_response.raise_for_status()
        robots = urllib.robotparser.RobotFileParser()
        robots.parse(robots_response.text.splitlines())

        urls = discover_article_urls(http)
        if args.limit:
            urls = urls[: args.limit]
        print(f"Found {len(urls)} article URLs")

        for number, url in enumerate(urls, start=1):
            if not robots.can_fetch(USER_AGENT, url):
                skipped.append(f"{url} (disallowed by robots.txt)")
                continue
            try:
                response = http.get(url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                skipped.append(f"{url} ({error})")
                continue

            article = parse_article(url, response.text)
            points = build_points(article) if article else []
            if not points:
                skipped.append(f"{url} (no content parsed)")
                continue

            qdrant.upsert(collection_name=args.collection, points=points, wait=True)
            total_chunks += len(points)
            print(f"[{number}/{len(urls)}] {article.title}: {len(points)} chunks")
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Indexed {total_chunks} chunks into '{args.collection}' at {args.qdrant_url}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
