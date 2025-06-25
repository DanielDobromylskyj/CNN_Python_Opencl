from concurrent.futures import ThreadPoolExecutor, as_completed

def enqueue_many(func, argss, show_progress_text=None, max_workers=16):
    results = [None] * len(argss)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, *args): i for i, args in enumerate(argss)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            completed += 1
            if show_progress_text:
                print(f"\r{show_progress_text} ({completed} / {len(results)})", end="")

    return results