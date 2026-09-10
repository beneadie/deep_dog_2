"""Reddit tools (OAuth): search, browse, batch-read posts, user credibility.

All read/search calls go through Reddit's OAuth API (oauth.reddit.com).
Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME,
REDDIT_PASSWORD in .env.

The batch read tool (get_reddit_posts) is a stub intercepted by the platform
agent engine's tool_node.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Literal

import requests
from langchain_core.tools import tool

from deep_research.secrets import get_secret

logger = logging.getLogger(__name__)

_DARK_ORANGE = "\033[38;5;166m"
_RESET = "\033[0m"

REDDIT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 "
    "Safari/537.36 RedditAgent/1.0"
)

# ── OAuth flow ─────────────────────────────────────────────────────────

_reddit_token_cache: dict = {"token": None, "expires_at": 0}

# Guards token fetch/refresh. Must only be touched on the event-loop thread
# (asyncio.Lock is not thread-safe), so the fetch itself runs via to_thread
# while the lock is held here on the loop.
_reddit_token_lock = asyncio.Lock()


def _get_reddit_token() -> str:
    """Obtain or refresh a Reddit OAuth token via password grant (TTL cached)."""
    now = time.time()
    if _reddit_token_cache["token"] and now < _reddit_token_cache["expires_at"]:
        return _reddit_token_cache["token"]

    client_id = get_secret("REDDIT_CLIENT_ID") or ""
    client_secret = get_secret("REDDIT_CLIENT_SECRET") or ""
    username = get_secret("REDDIT_USERNAME") or ""
    password = get_secret("REDDIT_PASSWORD") or ""

    if not all([client_id, client_secret, username, password]):
        raise ValueError(
            "Reddit OAuth credentials missing. Set REDDIT_CLIENT_ID, "
            "REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD in .env"
        )

    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "password", "username": username, "password": password}
    headers = {"User-Agent": REDDIT_USER_AGENT}
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth, data=data, headers=headers, timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            _reddit_token_cache["token"] = result["access_token"]
            _reddit_token_cache["expires_at"] = now + result.get("expires_in", 86400) - 60
            return _reddit_token_cache["token"]
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)
    raise last_error


def _oauth_headers() -> dict:
    """Return headers with Bearer token for oauth.reddit.com calls."""
    return {"Authorization": f"Bearer {_get_reddit_token()}", "User-Agent": REDDIT_USER_AGENT}


async def _ensure_reddit_token() -> None:
    """Ensure a fresh token is cached before dispatch to an executor thread.

    Runs entirely on the event-loop thread; the blocking OAuth POST happens
    in a worker thread via asyncio.to_thread while the lock is held. This
    serializes concurrent first-calls so exactly one token fetch occurs.
    """
    now = time.time()
    if _reddit_token_cache["token"] and now < _reddit_token_cache["expires_at"]:
        return
    async with _reddit_token_lock:
        now = time.time()
        if _reddit_token_cache["token"] and now < _reddit_token_cache["expires_at"]:
            return
        await asyncio.to_thread(_get_reddit_token)


async def _reddit_get(url: str, params: dict = None):
    """Sync requests call wrapped in run_in_executor — keeps blocking IO off the loop."""
    await _ensure_reddit_token()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: requests.get(url, headers=_oauth_headers(), params=params, timeout=15),
    )


def _oauth_url(url: str) -> str:
    """Convert a reddit.com URL to its oauth.reddit.com equivalent."""
    return url.replace("https://www.reddit.com", "https://oauth.reddit.com") \
              .replace("https://reddit.com", "https://oauth.reddit.com") \
              .rstrip("/").removesuffix(".json")


def _post_date_and_age(created_utc: float) -> tuple[str, str]:
    """Return an absolute post date and a human-readable relative age."""
    if not created_utc:
        return "unknown", "unknown"

    age_seconds = max(0, time.time() - created_utc)
    if age_seconds < 60:
        age = "just now"
    elif age_seconds < 3600:
        age = f"{int(age_seconds // 60)} min ago"
    elif age_seconds < 86400:
        hours = int(age_seconds // 3600)
        age = f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif age_seconds < 604800:
        days = int(age_seconds // 86400)
        age = f"{days} day{'s' if days != 1 else ''} ago"
    elif age_seconds < 2592000:
        weeks = int(age_seconds // 604800)
        age = f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif age_seconds < 31536000:
        months = int(age_seconds // 2592000)
        age = f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = int(age_seconds // 31536000)
        age = f"{years} year{'s' if years != 1 else ''} ago"

    return datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d"), age


def _reddit_search_item(post: dict) -> dict:
    """Build the indexed metadata retained for a Reddit search result."""
    return {
        "id": post["url"],
        "title": post["title"],
        "url": post["url"],
        "date": post["date"],
        "age": post["age"],
        "score": post["score"],
        "comments": post["comments"],
    }


def _format_reddit_comments(
    comments: list,
    depth: int = 0,
    parent_author: str | None = None,
) -> list[str]:
    """Format comments recursively while preserving reply relationships."""
    lines = []
    for comment in comments:
        if comment.get("kind") != "t1":
            continue
        data = comment.get("data", {})
        author = data.get("author", "[deleted]")
        body = data.get("body", "[removed]")
        score = data.get("score", 0)
        indent = "  " * depth
        reply_info = f" (replying to u/{parent_author})" if parent_author and depth > 0 else ""
        lines.append(f"{indent}**u/{author}** [{score:+d}]{reply_info}:")
        for body_line in body.split("\n"):
            lines.append(f"{indent}> {body_line}")
        lines.append("")

        replies = data.get("replies")
        if replies and isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            lines.extend(_format_reddit_comments(reply_children, depth + 1, author))
    return lines


def _format_reddit_posts(posts: list, header: str) -> str:
    if not posts:
        return header + "\n(no posts)"
    lines = [header, "", f"{'Date':<12} | {'Age':>14} | {'Score':>6} | {'Cmts':>5} | Title"]
    lines.append("-" * 110)
    for p in posts:
        lines.append(f"{p['date']:<12} | {p['age']:>14} | {p['score']:>6} | {p['comments']:>5} | {p['title'][:60]}")
        lines.append(f"    URL: {p['url']}")
        lines.append("")
    return "\n".join(lines)


# ── Search / browse tools ──────────────────────────────────────────────

@tool(parse_docstring=True)
async def search_term_in_subreddit(
    query: str,
    sort: Literal["relevance", "new", "top", "comments"] = "relevance",
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "year",
    limit: int = 100,
    subreddit: str = "",
) -> str:
    """Search Reddit by keyword — either across all of Reddit or within one subreddit.

     This is your PRIMARY Reddit discovery tool. Leave `subreddit` empty to
     search all of Reddit; pass a subreddit (e.g. 'stocks') to restrict the
     search to that community. Results include the post date and relative age,
     and carry an [S#] handle — read promising
    posts with get_reddit_posts(items=[{"ref": "S1", "index": N}]) and save
    them with batch_save_selected.

    Use it strategically:
    - Run MULTIPLE searches in parallel in one turn (one call per angle) to
      cover the topic broadly — e.g. one general search plus one per relevant
      subreddit, or several alternative phrasings.
    - sort: 'relevance' = best match (default); 'comments' = most-discussed
      threads (best for gauging real community opinion); 'new' = most recent;
      'top' = highest-scored.
    - time_filter: 'year' for recent discussion (a good default); 'month' or
      'week' for fast-moving topics; 'all' for evergreen questions; 'hour'/'day'
      only for genuinely live events.
    - Query tips: quote exact phrases ("opencode go"), join alternatives with
      OR, and vary synonyms across your parallel searches to widen coverage.
    - Each call counts toward your SEARCH budget — do not re-search the same
      angle once capped; read what you already found instead.

    Args:
        query: Search query (e.g. 'NVIDIA earnings', 'Tesla OR TSLA')
        sort: Sort order — 'relevance', 'new', 'top', or 'comments'
        time_filter: Time window — 'hour', 'day', 'week', 'month', 'year', 'all'
        limit: Max posts to fetch (1-200, paginated automatically)
        subreddit: Optional subreddit to restrict search to (e.g. 'stocks')
    """
    limit = min(max(1, limit), 200)
    if subreddit:
        subreddit = subreddit.lower().replace("r/", "").strip()
        base_url = _oauth_url(f"https://oauth.reddit.com/r/{subreddit}/search")
    else:
        base_url = _oauth_url("https://oauth.reddit.com/search")

    params = {"q": query, "restrict_sr": "on" if subreddit else "off",
              "limit": min(100, limit), "sort": sort, "t": time_filter}

    logger.info(f"{_DARK_ORANGE}Search Reddit{_RESET}: '{query}' (sort={sort}, time={time_filter}, limit={limit})")

    all_posts = []
    after_token = None
    try:
        while len(all_posts) < limit:
            if after_token:
                params["after"] = after_token
            resp = await _reddit_get(base_url, params)
            if resp.status_code == 429:
                logger.warning("Reddit rate limited (429); returning partial results")
                break
            if resp.status_code != 200:
                break
            data = resp.json()
            posts_raw = data.get("data", {}).get("children", [])
            after_token = data.get("data", {}).get("after")
            for post in posts_raw:
                p = post.get("data", {})
                date, age = _post_date_and_age(p.get("created_utc", 0))
                all_posts.append({
                    "title": p.get("title", ""),
                    "score": p.get("score", 0),
                    "comments": p.get("num_comments", 0),
                    "date": date,
                    "age": age,
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                })
            if not after_token or len(all_posts) >= limit:
                break
            await asyncio.sleep(1)
    except Exception as e:
        return f"Error searching Reddit: {e}"

    if not all_posts:
        return {"display": f"No results for: '{query}'", "items": []}
    display = _format_reddit_posts(all_posts[:limit], f"Found {len(all_posts[:limit])} posts for '{query}':")
    return {
        "display": display,
        "items": [_reddit_search_item(p) for p in all_posts[:limit]],
    }


@tool(parse_docstring=True)
async def get_subreddit_posts(
    subreddit: str,
    listing: Literal["hot", "new", "top"] = "hot",
    limit: int = 100,
) -> str:
    """Fetch a feed of posts from a specific subreddit (no search term needed).

     Use this to discover what's currently trending in a community when you have
     no specific query — e.g. scan r/opencode to see the latest sentiment.
     Results include the post date and relative age, and carry an [S#] handle;
     read posts with get_reddit_posts and save
    them with batch_save_selected.

    - listing: 'hot' = currently popular (good default for live sentiment);
      'new' = most recent (for breaking events); 'top' = all-time best.
    - This counts toward your SEARCH budget, so use it to complement
      search_term_in_subreddit rather than over-using either.
    - No time filter is available here — use search_term_in_subreddit's
      time_filter if you need to constrain by recency.

    Args:
        subreddit: Subreddit name (e.g. 'stocks', 'investing', 'nvidia')
        listing: Feed type — 'hot', 'new', or 'top'
        limit: Max posts to fetch (1-200, paginated automatically)
    """
    subreddit = subreddit.lower().replace("r/", "").strip()
    limit = min(max(1, limit), 200)
    base_url = _oauth_url(f"https://oauth.reddit.com/r/{subreddit}/{listing}")

    logger.info(f"{_DARK_ORANGE}Reddit{_RESET} fetching {limit} {listing} posts from r/{subreddit}")

    all_posts = []
    after_token = None
    try:
        while len(all_posts) < limit:
            params = {"limit": min(50, limit - len(all_posts))}
            if after_token:
                params["after"] = after_token
            resp = await _reddit_get(base_url, params)
            if resp.status_code != 200:
                break
            data = resp.json()
            posts_raw = data.get("data", {}).get("children", [])
            after_token = data.get("data", {}).get("after")
            for post in posts_raw:
                p = post.get("data", {})
                date, age = _post_date_and_age(p.get("created_utc", 0))
                all_posts.append({
                    "title": p.get("title", ""),
                    "score": p.get("score", 0),
                    "comments": p.get("num_comments", 0),
                    "date": date,
                    "age": age,
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                })
            if not after_token or len(all_posts) >= limit:
                break
            await asyncio.sleep(0.5)
    except Exception as e:
        return f"Error fetching r/{subreddit}: {e}"

    if not all_posts:
        return {"display": f"No posts found in r/{subreddit}.", "items": []}
    display = _format_reddit_posts(all_posts[:limit], f"Found {len(all_posts[:limit])} {listing} posts in r/{subreddit}:")
    return {
        "display": display,
        "items": [_reddit_search_item(p) for p in all_posts[:limit]],
    }


async def _fetch_reddit_post(url: str, include_comments: bool = True) -> str:
    """Fetch a single Reddit post (post body + optional full comment tree)."""
    logger.info(f"{_DARK_ORANGE}Read Reddit post{_RESET}: {url[:80]}...")
    try:
        json_url = _oauth_url(url) + ".json"
        resp = await _reddit_get(json_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error reading Reddit post: {e}"

    if not isinstance(data, list) or len(data) < 2:
        return f"Unexpected Reddit response for: {url}"

    post_data = data[0]["data"]["children"][0]["data"]
    comments_data = data[1]["data"]["children"]

    lines = [
        f"# {post_data.get('title', '[No Title]')}",
        f"**Author:** u/{post_data.get('author', '[deleted]')}  "
        f"| **Score:** {post_data.get('score', 0)}  "
        f"| **Comments:** {post_data.get('num_comments', 0)}",
        "",
    ]
    selftext = post_data.get("selftext", "")
    if selftext:
        lines.append("## Post Content")
        lines.append(selftext)
        lines.append("")

    if post_data.get("is_self") is False and post_data.get("url"):
        lines.append(f"**External link:** {post_data.get('url')}")
        lines.append("")

    if not include_comments:
        lines.append("")
        lines.append("(Comments excluded — include_comments=False)")
        lines.append("Save this post with batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}]) — use the ref+index from the header above.")
        return "\n".join(lines)

    lines.append("---")
    lines.append("## All Comments")
    lines.append("")

    lines.extend(_format_reddit_comments(comments_data))

    logger.info(f"{_DARK_ORANGE}Reddit post{_RESET} read "
                f"({post_data.get('num_comments', 0)} comments)")
    lines.append("")
    lines.append("Save this post with batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}]) — use the ref+index from the header above.")
    return "\n".join(lines)


@tool(parse_docstring=True)
async def get_reddit_posts(items: list, include_comments: bool = True) -> str:
    """Read one or more Reddit posts (full body + attributed comment thread) in one call.

    Pass posts by their [S#] ref + 1-based index from a search or feed result,
    e.g. get_reddit_posts(items=[{"ref": "S1", "index": 2}, ...]). Read up to 8
    posts per call. Keep include_comments=True — the comment thread is where the
    real discussion and disagreement live, which is the whole point of Reddit.
    Nested replies identify the author they directly answer.

    CHOOSE WHICH POSTS TO READ (default priorities — unless the user says otherwise):
    - High comment counts (more discussion → more diverse viewpoints)
    - Controversial or actively-debated topics (genuine disagreement, not echo chambers)
    - Specific data, numbers, or detailed analysis
    - Expert / insider perspectives (professionals weighing in)
    - Contrarian views that challenge the mainstream (consider, don't blindly adopt)

    AVOID (default): generic "daily discussion" threads, threads with <10 comments,
    meme/joke threads, and near-duplicate posts (read the better one).

    Read the most promising items FIRST, immediately after each search batch,
    before searching again. In curation mode, read broadly across your budget
    and save the best posts near the END with batch_save_selected (you may read
    and save in the same turn — reads run before saves).

    Args:
        items: List of objects, each {"ref": "S1", "index": 2}.
        include_comments: Include the full comment thread (default True).
    """
    return ""  # state write handled by tool_node


@tool(parse_docstring=True)
async def search_subreddits(query: str, limit: int = 10) -> str:
    """Discover subreddit community names by topic.

    Use only as a FALLBACK when you don't already know the relevant subreddit
    names (most topics have obvious communities). Returns subreddit names,
    subscriber counts, and descriptions — then explore them with
    search_term_in_subreddit or get_subreddit_posts. Counts toward your
    SEARCH budget, so don't over-use it.

    Args:
        query: Topic to search for (e.g. 'machine learning', 'electric vehicles')
        limit: Max subreddits to return (1-25)
    """
    limit = min(max(1, limit), 25)
    base_url = _oauth_url("https://oauth.reddit.com/subreddits/search")
    params = {"q": query, "limit": limit}

    logger.info(f"{_DARK_ORANGE}Search subreddits{_RESET}: '{query}'")
    try:
        resp = await _reddit_get(base_url, params)
        if resp.status_code != 200:
            return f"Subreddit search failed (HTTP {resp.status_code})"
        data = resp.json()
    except Exception as e:
        return f"Error searching subreddits: {e}"

    subs = data.get("data", {}).get("children", [])
    if not subs:
        return f"No subreddits found for: '{query}'"
    lines = [f"Found {len(subs)} subreddits for '{query}':", ""]
    for s in subs:
        d = s.get("data", {})
        nsfw = " [NSFW]" if d.get("over18", False) else ""
        lines.append(f"r/{d.get('display_name', '?')}{nsfw}  "
                     f"({(d.get('subscribers') or 0):,} subscribers)")
        desc = (d.get("public_description") or "")[:150]
        if desc:
            lines.append(f"  {desc}")
        lines.append(f"  https://reddit.com{d.get('url', '')}")
        lines.append("")
    lines.append("Use get_subreddit_posts or search_term_in_subreddit to explore these communities.")
    return "\n".join(lines)


@tool(parse_docstring=True)
async def check_user_profile(username: str) -> str:
    """Look up a Reddit user's profile — account age, karma, moderator status.

    Use before citing a specific user as an authority: a brand-new account with
    low karma or no history is a weak source, while a long-lived, high-karma
    account (ideally a verified moderator/employee) is much stronger. Does NOT
    count toward your search or read budget.

    Args:
        username: Reddit username (without u/ prefix)
    """
    username = username.lstrip("u/")
    url = _oauth_url(f"https://oauth.reddit.com/user/{username}/about")
    logger.info(f"{_DARK_ORANGE}Check user{_RESET}: u/{username}")
    try:
        resp = await _reddit_get(url)
        if resp.status_code == 404:
            return f"User u/{username} not found (deleted or suspended)."
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error fetching user profile: {e}"

    d = data.get("data", {})
    created_ts = d.get("created_utc", 0)
    age_days = (time.time() - created_ts) / 86400 if created_ts else 0
    return (
        f"**u/{d.get('name', username)}**\n"
        f"  Account age: {age_days:.0f} days  |  "
        f"Link karma: {d.get('link_karma', 0):,}  |  "
        f"Comment karma: {d.get('comment_karma', 0):,}\n"
        f"  Verified email: {d.get('has_verified_email', False)}  |  "
        f"Moderator: {d.get('is_mod', False)}  |  "
        f"Employee: {d.get('is_employee', False)}\n"
        f"  Profile: https://reddit.com/u/{d.get('name', username)}"
    )
