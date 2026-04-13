import requests
import time
import pandas as pd
from datetime import datetime, timedelta
import argparse

BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"


def fetch_subreddit_posts(
    subreddit=None,
    after=None,
    before=None,
    max_posts=5000,
    sleep_seconds=0.5,
    min_delay_seconds=0,
):
    all_posts = []
    seen_ids = set()

    params = {
        "after": after,
        "before": before,
        "limit": 100,
        "sort": "asc",
    }

    if subreddit:
        params["subreddit"] = subreddit

    while len(all_posts) < max_posts:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            break

        added_this_page = 0

        for post in data:
            if len(all_posts) >= max_posts:
                break

            post_id = post.get("id")
            if not post_id or post_id in seen_ids:
                continue

            created_utc = post.get("created_utc")
            retrieved_on = post.get("retrieved_on")

            if created_utc is None or retrieved_on is None:
                continue

            age_seconds = retrieved_on - created_utc
            if age_seconds < min_delay_seconds:
                continue

            seen_ids.add(post_id)

            all_posts.append({
                "id": post_id,
                "subreddit": post.get("subreddit"),
                "title": post.get("title"),
                "score": post.get("score"),
                "ups": post.get("ups"),
                "upvote_ratio": post.get("upvote_ratio"),
                "num_comments": post.get("num_comments"),
                "created_utc": created_utc,
                "retrieved_on": retrieved_on,
                "age_seconds": age_seconds,
                "selftext": post.get("selftext") or "",
                "author": post.get("author"),
                "is_self": post.get("is_self"),
                "is_video": post.get("is_video"),
                "post_hint": post.get("post_hint"),
                "domain": post.get("domain"),
                "link_flair_text": post.get("link_flair_text"),
                "over_18": post.get("over_18"),
                "spoiler": post.get("spoiler"),
                "stickied": post.get("stickied"),
                "locked": post.get("locked"),
                "subreddit_subscribers": post.get("subreddit_subscribers"),
                "thumbnail": post.get("thumbnail"),
                "thumbnail_height": post.get("thumbnail_height"),
                "thumbnail_width": post.get("thumbnail_width"),
                "removed_by_category": post.get("removed_by_category"),
                "url": post.get("url"),
            })

            added_this_page += 1

        if added_this_page == 0:
            break

        last_utc = data[-1].get("created_utc")
        if last_utc is None:
            break

        params["after"] = last_utc + 1

        scope = subreddit if subreddit else "ALL"
        print(f"[{scope}] {after} -> {before}: {len(all_posts)} posts")
        time.sleep(sleep_seconds)

    return pd.DataFrame(all_posts)


def generate_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    while current < end:
        next_day = current + timedelta(days=1)
        yield current.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")
        current = next_day


def run_collection(
    subreddits=None,
    start_date=None,
    end_date=None,
    max_posts_per_day=5000,
    output_file="reddit_data.csv",
    min_delay_seconds=0,
):
    all_data = []

    if subreddits:
        for subreddit in subreddits:
            print(f"\n=== {subreddit} ===")

            for after, before in generate_dates(start_date, end_date):
                print(f"\n--- {after} -> {before} ---")

                df = fetch_subreddit_posts(
                    subreddit=subreddit,
                    after=after,
                    before=before,
                    max_posts=max_posts_per_day,
                    min_delay_seconds=min_delay_seconds,
                )

                if not df.empty:
                    all_data.append(df)
    else:
        print("\n=== ALL SUBREDDITS ===")

        for after, before in generate_dates(start_date, end_date):
            print(f"\n--- {after} -> {before} ---")

            df = fetch_subreddit_posts(
                subreddit=None,
                after=after,
                before=before,
                max_posts=max_posts_per_day,
                min_delay_seconds=min_delay_seconds,
            )

            if not df.empty:
                all_data.append(df)

    if not all_data:
        print("No data collected")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.drop_duplicates(subset="id", inplace=True)

    final_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(final_df)} posts -> {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subreddits", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--max-posts-per-day", type=int, default=5000)
    parser.add_argument("--output-file", type=str, default="reddit_data.csv")
    parser.add_argument("--min-delay-seconds", type=int, default=0)

    return parser.parse_args()


def resolve_dates(start_date, end_date):
    if start_date and end_date:
        return start_date, end_date

    today = datetime.utcnow().date()
    default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=3)).strftime("%Y-%m-%d")

    return start_date or default_start, end_date or default_end


if __name__ == "__main__":
    args = parse_args()

    if args.subreddits:
        subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    else:
        subreddits = None

    start_date, end_date = resolve_dates(args.start_date, args.end_date)

    run_collection(
        subreddits=subreddits,
        start_date=start_date,
        end_date=end_date,
        max_posts_per_day=args.max_posts_per_day,
        output_file=args.output_file,
        min_delay_seconds=args.min_delay_seconds,
    )